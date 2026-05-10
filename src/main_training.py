import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
import wandb
from lightning import Fabric
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.loader import loaders_from_config
from models.s1 import FixedFilterFeatureExtractor
from models.s2_fragments import LateralNetwork
from utils.config import get_config
from utils.custom_print import print_start, print_warn
from utils.loggers import loggers_from_conf
from utils.store_load_run import load_run, save_run


def parse_args(parser: Optional[argparse.ArgumentParser] = None):
    """
    Parse command line arguments for training configuration.
    
    Parses configuration file path and optional training parameters like wandb logging,
    plotting, model saving/loading paths, and S1 filter learning mode.
    
    Args:
        parser (Optional[argparse.ArgumentParser]): Optional custom ArgumentParser instance.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Lateral Connections Stage 1")
    parser.add_argument("config",
                        type=str,
                        help="Path to the config file",
                        )
    parser.add_argument('--wandb',
                        action='store_true',
                        default=False,
                        dest='logging:wandb:active',
                        help='Log to wandb'
                        )
    parser.add_argument('--plot',
                        action='store_true',
                        default=False,
                        dest='run:plots:enable',
                        help='Plot results'
                        )
    parser.add_argument('--store',
                        type=str,
                        dest='run:store_state_path',
                        help='Path where the model will be stored'
                        )
    parser.add_argument('--load',
                        type=str,
                        dest='run:load_state_path',
                        help='Path from where the model will be loaded'
                        )
    parser.add_argument('--learn_s1',
                        action='store_true',
                        default=False,
                        dest='feature_extractor:learn_s1',
                        help='Whether to learn the S1 filters (instead of using fixed filters)'
                        )
    # parser.add_argument('--ts',
    #                     type=int,
    #                     default=5,
    #                     dest='lateral_model:max_timesteps',
    #                     help='Number of timesteps to train the lateral model for'
    #                     )

    args = parser.parse_args()
    return args


def configure(parser: Optional[argparse.ArgumentParser] = None) -> Dict[str, Optional[Any]]:
    """
    Load and configure the experiment from command line arguments.
    
    Parses arguments, loads configuration file, and sets up PyTorch backend for deterministic training.
    Warns if CUDA is unavailable for faster training.
    
    Args:
        parser (Optional[argparse.ArgumentParser]): Optional custom ArgumentParser instance.
    
    Returns:
        Dict[str, Optional[Any]]: Complete configuration dictionary.
    """
    args = parse_args(parser)
    config = get_config(args.config, args)
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        print_warn("CUDA is not available.", title="Slow training expected.")
    return config


def setup_fabric(config: Dict[str, Optional[Any]]) -> Fabric:
    """
    Initialize Lightning Fabric for distributed training and acceleration.
    
    Creates a Fabric instance with auto-detected accelerator, sets up loggers from config,
    launches training environment, and seeds random number generators for reproducibility.
    
    Args:
        config (Dict[str, Optional[Any]]): Configuration dictionary.
    
    Returns:
        Fabric: Configured Lightning Fabric instance.
    """
    loggers = loggers_from_conf(config)
    fabric = Fabric(accelerator="auto", devices=1, loggers=loggers, callbacks=[])
    fabric.launch()
    fabric.seed_everything(1)
    return fabric


def setup_wandb(config: Dict[str, Optional[Any]]) -> None:
    """
    Initialize Weights & Biases logging if enabled in configuration.
    
    Sets up wandb project, passes full configuration for tracking, and assigns job metadata
    including job type and experiment group if wandb logging is active.
    
    Args:
        config (Dict[str, Optional[Any]]): Configuration dictionary containing wandb settings.
    """
    if "wandb" in config['logging'].keys() and config['logging']['wandb']['active']:
        wandb_conf = config['logging']['wandb']
        wandb.init(project=wandb_conf['project'],
                   config=config,
                   job_type=wandb_conf['job_type'],
                   group=wandb_conf['group'])


def setup_dataloader(config: Dict[str, Optional[Any]], fabric: Fabric) -> tuple[DataLoader, DataLoader]:
    """
    Create and configure dataloaders for training and evaluation.
    
    Loads dataloaders from config, then wraps them with Fabric for proper distributed training
    and device placement.
    
    Args:
        config (Dict[str, Optional[Any]]): Configuration dictionary.
        fabric (Fabric): Initialized Lightning Fabric instance.
    
    Returns:
        tuple[DataLoader, DataLoader]: Training and evaluation dataloaders.
    """
    train_loader, eval_loader, _ = loaders_from_config(config)
    train_loader = fabric.setup_dataloaders(train_loader)
    eval_loader = fabric.setup_dataloaders(eval_loader)
    return train_loader, eval_loader


def setup_feature_extractor(config: Dict[str, Optional[Any]], fabric: Fabric) -> pl.LightningModule:
    """
    Initialize the feature extractor model for visual processing.
    
    Creates a FixedFilterFeatureExtractor (S1 stage) that extracts features from images
    before feeding them into the lateral network. Optionally learns S1 filters if enabled in config.
    
    Args:
        config (Dict[str, Optional[Any]]): Configuration dictionary.
        fabric (Fabric): Initialized Lightning Fabric instance.
    
    Returns:
        pl.LightningModule: Feature extractor model wrapped for training.
    """
    feature_extractor = FixedFilterFeatureExtractor(config, fabric)
    feature_extractor = fabric.setup(feature_extractor)
    return feature_extractor


def cycle(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        batch: Tensor,
        store_tensors: Optional[bool] = False,
        mode: Optional[str] = "train",
) -> Optional[tuple[Tensor, Tensor, Tensor, Tensor]]:
    """
    Execute a single forward pass through feature extraction and lateral network processing.
    
    Extracts features using the feature extractor, then iteratively processes them through
    the lateral network across multiple timesteps. Applies Hebbian updates during training.
    Optionally stores intermediate tensors for visualization and analysis.
    
    Args:
        config (Dict[str, Optional[Any]]): Configuration dictionary.
        feature_extractor (pl.LightningModule): Feature extractor model.
        lateral_network (LateralNetwork): Lateral connections network model.
        batch (Tensor): Input images batch.
        store_tensors (Optional[bool]): Whether to store and return intermediate tensors. Defaults to False.
        mode (Optional[str]): Execution mode, either "train" or "eval". Defaults to "train".
    
    Returns:
        Optional[tuple[Tensor, Tensor, Tensor, Tensor]]: When store_tensors=True, returns tuple of
            (extracted_features, input_features, lateral_features, lateral_features_float).
            Returns None when store_tensors=False.
    """
    assert mode in ["train", "eval"], "Mode must be either train or eval"

    with torch.no_grad():
        features = feature_extractor(batch)

    z = None

    input_features, lateral_features, lateral_features_f, l2_features, l2h_features = [], [], [], [], []
    for view_idx in range(features.shape[1]):
        x_view_features = features[:, view_idx, ...]

        if store_tensors:
            input_features.append(x_view_features)

        if z is None:
            z = torch.zeros((x_view_features.shape[0], lateral_network.model.out_channels, x_view_features.shape[2],
                             x_view_features.shape[3]), device=batch.device)

        features_lat, features_lat_float = [], []
        for t in range(config["lateral_model"]["max_timesteps"]):
            lateral_network.model.update_ts(t)
            x_in = torch.cat([x_view_features, z], dim=1)
            z_float, z = lateral_network(x_in)

            features_lat.append(z)
            if store_tensors:
                features_lat_float.append(z_float)

        features_lat = torch.stack(features_lat, dim=1)
        features_lat_median = torch.median(features_lat, dim=1)[0]
        if store_tensors:
            features_lat_float = torch.stack(features_lat_float, dim=1)

        if mode == "train":
            # Train at the end after all timesteps (use median activation per cell),
            x_rearranged = lateral_network.model.s2.rearrange_input(
                torch.cat([x_view_features, features_lat_median], dim=1))
            lateral_network.model.s2.hebbian_update(x_rearranged, features_lat_median)

        if store_tensors:
            features_lat_float_median = torch.median(features_lat_float, dim=1)[0]
            features_lat = torch.cat([features_lat, features_lat_median.unsqueeze(1)], dim=1)
            features_lat_float = torch.cat([features_lat_float, features_lat_float_median.unsqueeze(1)], dim=1)
            lateral_features.append(features_lat)
            lateral_features_f.append(features_lat_float)

    if store_tensors:
        return features, torch.stack(input_features, dim=1), torch.stack(lateral_features, dim=1), torch.stack(
            lateral_features_f, dim=1)


def single_train_epoch(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        train_loader: DataLoader,
        epoch: int,
) -> None:
    """
    Execute a single training epoch over the full training dataset.
    
    Processes all batches through the model in train mode, applying Hebbian weight updates
    during lateral network processing. Progress is displayed via progress bar.
    
    Args:
        config (Dict[str, Optional[Any]]): Configuration dictionary.
        feature_extractor (pl.LightningModule): Feature extractor model.
        lateral_network (LateralNetwork): Lateral network model.
        train_loader (DataLoader): Training dataset loader.
        epoch (int): Current epoch number.
    """
    feature_extractor.eval()
    lateral_network.eval()
    for i, batch in tqdm(enumerate(train_loader),
                         total=len(train_loader),
                         colour="GREEN",
                         desc=f"Train Epoch {epoch}/{config['run']['n_epochs']}"):
        cycle(config, feature_extractor, lateral_network, batch[0], store_tensors=False, mode="train")


def single_eval_epoch(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        test_loader: DataLoader,
        epoch: int,
) -> None:
    """
    Evaluate the model on test set and optionally generate visualizations.
    
    Processes test data in eval mode, collects activations, and generates plots if enabled.
    Logs plots to wandb if configured. Handles conditional plotting based on epoch and config settings.
    
    Args:
        config (Dict[str, Optional[Any]]): Configuration dictionary.
        feature_extractor (pl.LightningModule): Feature extractor model.
        lateral_network (LateralNetwork): Lateral network model.
        test_loader (DataLoader): Test dataset loader.
        epoch (int): Current epoch number.
    """
    feature_extractor.eval()
    lateral_network.eval()
    plt_img, plt_features, plt_input_features, plt_activations, plt_activations_f = [], [], [], [], []
    for i, batch in tqdm(enumerate(test_loader),
                         total=len(test_loader),
                         colour="GREEN",
                         desc=f"Testing Epoch {epoch}/{config['run']['n_epochs']}"):
        with torch.no_grad():
            features, input_features, lateral_features, lateral_features_f = cycle(config,
                                                                                   feature_extractor,
                                                                                   lateral_network,
                                                                                   batch[0],
                                                                                   store_tensors=True,
                                                                                   mode="eval")
            plt_img.append(batch[0])
            plt_features.append(features)
            plt_input_features.append(input_features)
            plt_activations.append(lateral_features)
            plt_activations_f.append(lateral_features_f)

    plot = config['run']['plots']['enable'] and \
           (not config['run']['plots']['only_last_epoch'] or epoch == config['run']['n_epochs'])
    wandb_b = config['logging']['wandb']['active']
    store_plots = config['run']['plots'].get('store_path', False)

    assert not wandb_b or (wandb_b and store_plots), "Wandb logging requires storing the plots."

    if plot:
        if epoch == 0:
            feature_extractor.plot_model_weights(show_plot=plot)
        plots_fp = lateral_network.plot_samples(plt_img,
                                                plt_features,
                                                plt_input_features,
                                                plt_activations,
                                                plt_activations_f,
                                                plot_input_features=epoch == 0,
                                                show_plot=plot)
        weights_fp = lateral_network.plot_model_weights(show_plot=plot)

        if wandb_b:
            logs = {str(pfp.name[:-4]): wandb.Image(str(pfp)) for pfp in plots_fp}
            logs |= {str(wfp.name[:-4]): wandb.Image(str(wfp)) for wfp in weights_fp}
            wandb.log(logs | {"epoch": epoch, "trainer/global_step": epoch})


def train(
        config: Dict[str, Optional[Any]],
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        train_loader: DataLoader,
        test_loader: DataLoader,
) -> None:
    """
    Main training loop over all epochs.
    
    Iterates through specified number of epochs, running training and evaluation phases.
    Logs metrics and saves checkpoints. Evaluates on initial epoch if plotting or wandb logging enabled.
    
    Args:
        config (Dict[str, Optional[Any]]): Configuration dictionary.
        feature_extractor (pl.LightningModule): Feature extractor module.
        lateral_network (LateralNetwork): Lateral network module.
        train_loader (DataLoader): Training dataset loader.
        test_loader (DataLoader): Test dataset loader.
    """
    start_epoch = config['run']['current_epoch']

    if config['logging']['wandb']['active'] or config['run']['plots']['enable']:
        single_eval_epoch(config, feature_extractor, lateral_network, test_loader, 0)
        lateral_network.on_epoch_end()  # print logs

    for epoch in range(start_epoch, config['run']['n_epochs']):
        single_train_epoch(config, feature_extractor, lateral_network, train_loader, epoch + 1)
        single_eval_epoch(config, feature_extractor, lateral_network, test_loader, epoch + 1)
        lateral_network.on_epoch_end()
        config['run']['current_epoch'] = epoch + 1


def setup_lateral_network(config: Dict[str, Optional[Any]], fabric: Fabric) -> LateralNetwork:
    """
    Initialize the lateral network model with lateral connections.
    
    Creates LateralNetwork instance and wraps it with Fabric for proper device placement
    and distributed training support.
    
    Args:
        config (Dict[str, Optional[Any]]): Configuration dictionary.
        fabric (Fabric): Initialized Lightning Fabric instance.
    
    Returns:
        LateralNetwork: Lateral network model wrapped with Fabric.
    """
    return fabric.setup(LateralNetwork(config, fabric))


def main() -> None:
    """
    Main entry point for the training script.
    
    Orchestrates the full training pipeline: loads configuration, initializes Fabric and models,
    sets up dataloaders, optionally loads checkpoints, and runs training with optional visualization
    and wandb logging. Saves final model state if configured.
    """
    print_start("Starting python script 'main_lateral_connections.py'...",
                title="Training S1: Lateral Connections Toy Example")
    config = configure()
    fabric = setup_fabric(config)
    train_loader, test_loader = setup_dataloader(config, fabric)
    feature_extractor = setup_feature_extractor(config, fabric)
    lateral_network = setup_lateral_network(config, fabric)

    if 'load_state_path' in config['run'] and config['run']['load_state_path'] != 'None':
        config, state = load_run(config, fabric)
        feature_extractor.load_state_dict(state['feature_extractor'])
        lateral_network.load_state_dict(state['lateral_network'])

    feature_extractor.eval()  # does not have to be trained
    if 'store_path' in config['run']['plots'] and config['run']['plots']['store_path'] is not None and \
            config['run']['plots']['store_path'] != 'None':
        fp = Path(config['run']['plots']['store_path'])
        if not fp.exists():
            fp.mkdir(parents=True, exist_ok=True)

    setup_wandb(config)
    train(config, feature_extractor, lateral_network, train_loader, test_loader)

    if 'store_state_path' in config['run'] and config['run']['store_state_path'] is not None and config['run'][
        'store_state_path'] != 'None':
        save_run(config, fabric,
                 components={'feature_extractor': feature_extractor, 'lateral_network': lateral_network})


if __name__ == '__main__':
    main()
