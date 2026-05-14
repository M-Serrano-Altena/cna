import argparse
from pathlib import Path
from typing import Any, Dict, Optional, cast

import lightning.pytorch as pl
import torch
import wandb
from lightning import Fabric
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from data.loader import loaders_from_config
from models.s1 import FixedFilterFeatureExtractor
from models.s2_fragments import LateralNetwork
from utils.config import get_config
from utils.custom_print import print_start, print_warn, print_logs
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


def setup_feature_extractor(config: Dict[str, Any], fabric: Fabric) -> tuple[pl.LightningModule, Optional[torch.optim.Optimizer]]:
    """
    Initialize the feature extractor model for visual processing.
    
    Creates a FixedFilterFeatureExtractor (S1 stage) that extracts features from images
    before feeding them into the lateral network. Optionally learns S1 filters if enabled in config.
    
    Args:
        config (Dict[str,Any]): Configuration dictionary.
        fabric (Fabric): Initialized Lightning Fabric instance.
    
    Returns:
        tuple[pl.LightningModule, Optional[torch.optim.Optimizer]]: Feature extractor model and optimizer wrapped for training.
    """
    default_feature_extractor_config = {
        'out_channels': 4,
        'learn_s1': False,
        's1_params': {
            'use_larger_weights': False,
            'threshold_f': 'threshold'
        },
        'lr': 1e-3
    }
    feature_extractor_config = config.get('feature_extractor', default_feature_extractor_config)
    if not isinstance(feature_extractor_config, dict):
        print_warn("Invalid feature extractor config. Using default fixed filters.", title="Config Warning")
        feature_extractor_config = default_feature_extractor_config
        
    # fixed feature extraction (non-learnable S1 filters)
    if not feature_extractor_config.get('learn_s1', default_feature_extractor_config['learn_s1']):
        feature_extractor = FixedFilterFeatureExtractor(config, fabric)
        feature_extractor = fabric.setup(feature_extractor)
        return feature_extractor, None
    
    # learnable S1 filters
    from models.s1 import AlternatingFeatureExtractor
    feature_extractor = AlternatingFeatureExtractor(config, fabric)
    optimizer = torch.optim.Adam(
        feature_extractor.parameters(),
        lr=feature_extractor_config.get('lr', default_feature_extractor_config['lr'])
    )
    feature_extractor, optimizer = fabric.setup(feature_extractor, optimizer)
    return feature_extractor, optimizer



def cycle(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        batch: Tensor,
        store_tensors: Optional[bool] = False,
        mode: Optional[str] = "train",
        s1_optimizer: Optional[torch.optim.Optimizer] = None,
        fixed_extractor: Optional[pl.LightningModule] = None
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
        s1_optimizer (Optional[torch.optim.Optimizer]): Optimizer for the S1 layer, used during training if learn_s1 is True. Defaults to None.
        fixed_extractor (Optional[pl.LightningModule]): Optional fixed feature extractor for evaluation mode when S1 is learnable. Defaults to None.
    
    Returns:
        Optional[tuple[Tensor, Tensor, Tensor, Tensor]]: When store_tensors=True, returns tuple of
            (extracted_features, input_features, lateral_features, lateral_features_float).
            Returns None when store_tensors=False.
    """
    assert mode in ["train", "eval"], "Mode must be either train or eval"

    learn_s1 = s1_optimizer is not None
    require_grad = learn_s1 and mode == "train"

    # Feature extraction: with grad only if we're learning S1
    with torch.set_grad_enabled(require_grad):
        features = feature_extractor(batch)

    z = None
    z_float = None

    input_features, lateral_features, lateral_features_f= [], [], []
    for view_idx in range(features.shape[1]):
        x_view_features = features[:, view_idx, ...]

        if store_tensors:
            input_features.append(x_view_features)

        if z is None:
            z = torch.zeros(
                (x_view_features.shape[0], lateral_network.model.out_channels,
                 x_view_features.shape[2], x_view_features.shape[3]),
                device=batch.device
            )


        features_lat, features_lat_float = [], []
        for t in range(config["lateral_model"]["max_timesteps"]):
            lateral_network.model.update_ts(t)
            x_in = torch.cat([x_view_features, z], dim=1)
            z_float, z = lateral_network(x_in, require_grad=require_grad)
            features_lat.append(z)
            if store_tensors:
                features_lat_float.append(z_float)

        features_lat = torch.stack(features_lat, dim=1)
        features_lat_median = torch.median(features_lat, dim=1)[0]
        if store_tensors:
            features_lat_float = torch.stack(features_lat_float, dim=1)

        if mode == "train":
            # S2 Hebbian update after all timesteps (use median activation per cell),
            x_rearranged = lateral_network.model.s2.rearrange_input(
                torch.cat([x_view_features, features_lat_median], dim=1))
            lateral_network.model.s2.hebbian_update(x_rearranged, features_lat_median)

            # S1 gradient update if learn_s1 is enabled:
            # maximize difference between active and inactive neurons using the final timestep activations as feedback
            if learn_s1 and z_float is not None:

                if fixed_extractor is not None:
                    with torch.no_grad():
                        target_logits = fixed_extractor.get_logits(batch)
                        target_logits_norm = F.normalize(target_logits, dim=1)

                    learned_logits = feature_extractor.get_logits(batch)
                    learned_logits_norm = F.normalize(learned_logits, dim=1)

                    s1_loss = F.mse_loss(learned_logits_norm, target_logits_norm)


                else:
                    active_mask = z > 0  # z is the binary output
                    if active_mask.any() and (~active_mask).any():
                        s1_loss = -(z_float[active_mask].mean() - z_float[~active_mask].mean())
                    else:
                        s1_loss = -z_float.mean()  # fallback if all active or all inactive
                
                l2 = 1e-4 * (feature_extractor.model.weight ** 2).mean()
                print_logs({"s1_loss": s1_loss.item(), "s1_weight_l2": l2})
                s1_loss = s1_loss + l2
                s1_optimizer.zero_grad()
                fabric.backward(s1_loss)
                s1_optimizer.step()
                if hasattr(feature_extractor, "log_step"):
                    feature_extractor.log_step(s1_loss.item(), z_float.detach(), z.detach())
            else:
                # Log S2 coherence even for fixed S1 to track S2 evolution
                if hasattr(feature_extractor, "log_step"):
                    feature_extractor.log_step(torch.tensor(0.0), z_float.detach(), z.detach())


        if store_tensors:
            features_lat_float_median = torch.median(features_lat_float, dim=1)[0]
            features_lat = torch.cat([features_lat, features_lat_median.unsqueeze(1)], dim=1)
            features_lat_float = torch.cat([features_lat_float, features_lat_float_median.unsqueeze(1)], dim=1)
            lateral_features.append(features_lat)
            lateral_features_f.append(features_lat_float)

    if store_tensors:
        return (
            features,
            torch.stack(input_features, dim=1),
            torch.stack(lateral_features, dim=1),
            torch.stack(lateral_features_f, dim=1),
        )


def single_train_epoch(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        train_loader: DataLoader,
        epoch: int,
        s1_optimizer: Optional[torch.optim.Optimizer] = None,
        fixed_extractor: Optional[pl.LightningModule] = None
) -> None:
    """
    Execute a single training epoch over the full training dataset.
    
    Processes all batches through the model in train mode, applying Hebbian weight updates
    during lateral network processing. Progress is displayed via progress bar.
    
    Args:
        config (Dict[str, Optional[Any]]): Configuration dictionary.
        fabric (Fabric): Initialized Lightning Fabric instance.
        feature_extractor (pl.LightningModule): Feature extractor model.
        lateral_network (LateralNetwork): Lateral network model.
        train_loader (DataLoader): Training dataset loader.
        epoch (int): Current epoch number.
        s1_optimizer (Optional[torch.optim.Optimizer]): Optimizer for S1 layer updates.
        fixed_extractor (Optional[pl.LightningModule]): Optional fixed feature extractor for evaluation mode when S1 is learnable. Defaults to None.
    """
    learn_s1 = s1_optimizer is not None
    if learn_s1:
        feature_extractor.train()
    else:
        feature_extractor.eval()

    lateral_network.eval()
    for i, batch in tqdm(enumerate(train_loader),
                         total=len(train_loader),
                         colour="GREEN",
                         desc=f"Train Epoch {epoch}/{config['run']['n_epochs']}"):
        cycle(config, fabric, feature_extractor, lateral_network, batch[0], store_tensors=False, mode="train", s1_optimizer=s1_optimizer, fixed_extractor=fixed_extractor)


def single_eval_epoch(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        test_loader: DataLoader,
        epoch: int,
        s1_optimizer: Optional[torch.optim.Optimizer] = None,
        fixed_extractor: Optional[pl.LightningModule] = None
) -> None:
    """
    Evaluate the model on test set and optionally generate visualizations.
    
    Processes test data in eval mode, collects activations, and generates plots if enabled.
    Logs plots to wandb if configured. Handles conditional plotting based on epoch and config settings.
    
    Args:
        config (Dict[str, Optional[Any]]): Configuration dictionary.
        fabric (Fabric): Initialized Lightning Fabric instance.
        feature_extractor (pl.LightningModule): Feature extractor model.
        lateral_network (LateralNetwork): Lateral network model.
        test_loader (DataLoader): Test dataset loader.
        epoch (int): Current epoch number.
        s1_optimizer (Optional[torch.optim.Optimizer]): Optimizer for S1 layer updates.
        fixed_extractor (Optional[pl.LightningModule]): Optional fixed feature extractor for evaluation mode when S1 is learnable. Defaults to None.
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
                                                                                   fabric,
                                                                                   feature_extractor,
                                                                                   lateral_network,
                                                                                   batch[0],
                                                                                   store_tensors=True,
                                                                                   mode="eval",
                                                                                   s1_optimizer=s1_optimizer,
                                                                                   fixed_extractor=fixed_extractor)
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



def single_train_epoch_s1_only(
    config: Dict[str, Optional[Any]],
    fabric: Fabric,
    feature_extractor: pl.LightningModule,
    fixed_extractor: pl.LightningModule,
    train_loader: DataLoader,
    epoch: int,
    s1_optimizer: torch.optim.Optimizer,
) -> None:
    feature_extractor.train()
    s1_loss = torch.tensor(0.0)
    l2 = torch.tensor(0.0)
    
    for i, batch in tqdm(enumerate(train_loader),
                         total=len(train_loader),
                         colour="GREEN",
                         desc=f"Train S1 Only Epoch {epoch}/{config['run']['n_epochs']}"):
        with torch.no_grad():
            target_logits = fixed_extractor.get_logits(batch[0])
            target_logits_norm = F.normalize(target_logits, dim=1)

        learned_logits = feature_extractor.get_logits(batch[0])
        learned_logits_norm = F.normalize(learned_logits, dim=1)

        s1_loss = F.mse_loss(learned_logits_norm, target_logits_norm)
        l2 = (feature_extractor.model.weight ** 2).mean()
        s1_loss = s1_loss + l2
        s1_optimizer.zero_grad()
        fabric.backward(s1_loss)
        s1_optimizer.step()

    print_logs({"s1_loss": s1_loss.item(), "s1_weight_l2": l2})
    



def train_s1_only(
    config: Dict[str, Optional[Any]],
    fabric: Fabric,
    feature_extractor: pl.LightningModule,
    fixed_extractor: pl.LightningModule,
    train_loader: DataLoader,
    s1_optimizer: torch.optim.Optimizer,
) -> None:
    start_epoch = config['run']['current_epoch']

    for epoch in range(start_epoch, config['run']['n_epochs']):
        single_train_epoch_s1_only(config, fabric, feature_extractor, fixed_extractor, train_loader, epoch + 1, s1_optimizer=s1_optimizer)
        
        if hasattr(feature_extractor, "get_and_reset_logs"):
            s1_logs = feature_extractor.get_and_reset_logs()
            feature_extractor.log_dict(s1_logs)
            print_logs(s1_logs)

        config['run']['current_epoch'] = epoch + 1

    # reset epoch to 0 for potential subsequent training phases (e.g. training S2 after S1)
    config['run']['current_epoch'] = start_epoch


def compare_s1_feature_extraction(
    learned_extractor: pl.LightningModule,
    fixed_extractor: pl.LightningModule,
    dataloader: DataLoader,
    n_batches: int = 5,
) -> float:
    """
    Compare learned S1 vs fixed S1 by computing MSE between normalized, flattened logits
    on the first `n_batches` from `dataloader`. Logs per-batch and average MSE.
    """
    learned_extractor.eval()
    fixed_extractor.eval()
    device = next(learned_extractor.parameters()).device
    total_mse = 0.0
    seen = 0
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        x = batch[0].to(device)
        with torch.no_grad():
            tgt = fixed_extractor.get_logits(x)
            src = learned_extractor.get_logits(x)
            # flatten per-sample (handles 4D or 5D logits) and normalize
            tgt_f = F.normalize(tgt.view(tgt.size(0), -1), dim=1)
            src_f = F.normalize(src.view(src.size(0), -1), dim=1)
            mse = F.mse_loss(src_f, tgt_f).item()
        print_logs({f"s1_compare_batch_{i}_mse": mse})
        total_mse += mse
        seen += 1
    avg = total_mse / seen if seen else 0.0
    print_logs({"s1_compare_mse_avg": avg})
    return avg


def train(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
        feature_extractor: pl.LightningModule,
        lateral_network: LateralNetwork,
        train_loader: DataLoader,
        test_loader: DataLoader,
        s1_optimizer: Optional[torch.optim.Optimizer] = None,
        fixed_extractor: Optional[pl.LightningModule] = None,
) -> None:
    """
    Main training loop over all epochs.
    
    Iterates through specified number of epochs, running training and evaluation phases.
    Logs metrics and saves checkpoints. Evaluates on initial epoch if plotting or wandb logging enabled.
    
    Args:
        config (Dict[str, Optional[Any]]): Configuration dictionary.
        fabric (Fabric): Initialized Lightning Fabric instance.
        feature_extractor (pl.LightningModule): Feature extractor module.
        lateral_network (LateralNetwork): Lateral network module.
        train_loader (DataLoader): Training dataset loader.
        test_loader (DataLoader): Test dataset loader.
        s1_optimizer (Optional[torch.optim.Optimizer]): Optimizer for S1 layer updates.
        fixed_extractor (Optional[pl.LightningModule]): Optional fixed feature extractor for evaluation mode when S1 is learnable. Defaults to None.
    """
    start_epoch = config['run']['current_epoch']

    if config['logging']['wandb']['active'] or config['run']['plots']['enable']:
        single_eval_epoch(config, fabric, feature_extractor, lateral_network, test_loader, 0)
        lateral_network.on_epoch_end()  # print logs

    for epoch in range(start_epoch, config['run']['n_epochs']):
        single_train_epoch(config, fabric, feature_extractor, lateral_network, train_loader, epoch + 1, s1_optimizer=s1_optimizer, fixed_extractor=fixed_extractor)
        single_eval_epoch(config, fabric, feature_extractor, lateral_network, test_loader, epoch + 1)
        
        lateral_network.on_epoch_end()
        if hasattr(feature_extractor, "get_and_reset_logs"):
            s1_logs = feature_extractor.get_and_reset_logs()
            feature_extractor.log_dict(s1_logs)
            print_logs(s1_logs)

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
    feature_extractor, s1_optimizer = setup_feature_extractor(config, fabric)
    lateral_network = setup_lateral_network(config, fabric)

    fixed_extractor = None
    if s1_optimizer is not None:
        config_fixed = deepcopy(config)
        config_fixed['feature_extractor']['learn_s1'] = False
        fixed_extractor, _ = setup_feature_extractor(config_fixed, fabric)

    if 'load_state_path' in config['run'] and config['run']['load_state_path'] != 'None':
        config, state = load_run(config, fabric)
        feature_extractor.load_state_dict(state['feature_extractor'])
        lateral_network.load_state_dict(state['lateral_network'])

    if hasattr(feature_extractor, "setup_logging"):
        init_weights = feature_extractor.model.weight.detach() if s1_optimizer is not None else None
        feature_extractor.setup_logging(init_weights)

    feature_extractor.eval()  # does not have to be trained
    if 'store_path' in config['run']['plots'] and config['run']['plots']['store_path'] is not None and \
            config['run']['plots']['store_path'] != 'None':
        fp = Path(config['run']['plots']['store_path'])
        if not fp.exists():
            fp.mkdir(parents=True, exist_ok=True)

    setup_wandb(config)

    # Train S1 filters alone for a few epochs if learn_s1 is enabled, to stabilize S1 representations before training S2 with Hebbian updates.
    # This can help prevent instability in early training when both S1 and S2 are learning simultaneously from random initializations.
    if s1_optimizer is not None and fixed_extractor is not None and config['feature_extractor'].get('learn_s1', False):
        train_s1_only(config, fabric, feature_extractor, fixed_extractor, train_loader, s1_optimizer)

    # train S2 while fixing S1
    train(config, fabric, feature_extractor, lateral_network, train_loader, test_loader, s1_optimizer=None, fixed_extractor=fixed_extractor)

    if 'store_state_path' in config['run'] and config['run']['store_state_path'] is not None and config['run'][
        'store_state_path'] != 'None':
        save_run(config, fabric,
                 components={'feature_extractor': feature_extractor, 'lateral_network': lateral_network})


if __name__ == '__main__':
    main()
