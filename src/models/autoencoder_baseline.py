from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim import Optimizer

from utils.meters import AverageMeter


class BaseLitModule(pl.LightningModule):
    """
    Lightning Base Module
    """

    def __init__(
            self,
            conf: Dict[str, Any],
            fabric: Fabric,
            logging_prefixes: Optional[List[str]] = None,
    ) -> None:
        """Initialize the Lightning base module.

        Args:
            conf (Dict[str, Any]): Configuration dictionary for the module.
            fabric (Fabric): lightning.fabric Fabric instance used to place tensors/models on devices.
            logging_prefixes (Optional[List[str]]): Optional list of prefixes used when logging
                (e.g. ["train", "val"]). If None, defaults to ["train", "val"].
        """
        super().__init__()
        self.conf = conf
        self.fabric = fabric
        if logging_prefixes is None:
            logging_prefixes = ["train", "val"]
        self.logging_prefixes = logging_prefixes
        self.avg_meters = {}
        self.current_epoch_ = 0

    def log_step(self,
                 logs: Dict[str, torch.Tensor],
                 prefix: Optional[str] = "",
                 ) -> None:
        """Update internal average meters with values from a training/validation step.

        Args:
            logs (Dict[str, torch.Tensor]): Mapping from metric name to tensor value to be
                accumulated in the average meters.
            prefix (Optional[str]): Optional prefix to prepend to metric names when storing
                meters (e.g. "train" or "val"). Defaults to empty string.
        """
        for k, v in logs.items():
            meter_name = f"{prefix}/{k}" if prefix != "" else f"{k}"
            if meter_name not in self.avg_meters:
                self.avg_meters[meter_name] = AverageMeter()
            self.avg_meters[meter_name](v)

    def log_(self) -> Dict[str, float]:
        """Aggregate averaged metrics, log them to Lightning and reset meters.

        Returns:
            Dict[str, float]: A dictionary containing the current epoch and the mean value for
                each tracked metric.
        """
        logs = {'epoch': self.current_epoch_}
        for m_name, m in self.avg_meters.items():
            val = m.mean
            if isinstance(val, torch.Tensor):
                val = val.item()
            logs[m_name] = val
            m.reset()
        self.log_dict(logs)
        return logs

    def epoch_end(self) -> Dict[str, float]:
        """Callback to be called at the end of an epoch.

        This aggregates and logs metrics via the method`log_` and advances the internal epoch
        counter.

        Returns:
            Dict[str, float]: The same dictionary returned by the method`log_`.
        """
        logs = self.log_()
        self.current_epoch_ += 1
        return logs


class Autoencoder(BaseLitModule):
    """
    Extract features from non-overlapping patches of an image using a VQ-VAE.
    """

    def __init__(self, conf: Dict[str, Any], fabric: Fabric):
        """Construct the Autoencoder Lightning module.

        Args:
            conf (Dict[str, Any]): Configuration dictionary containing dataset and
                optimizer parameters used by the module.
            fabric (Fabric): Fabric instance used to determine device placement.
        """
        super().__init__(conf, fabric, logging_prefixes=["train", "val"])
        self.model = self.configure_model()
        self.data_var = torch.mean(torch.Tensor(self.conf['dataset']['std'])).to(fabric.device) ** 2
        self.loss_f = nn.MSELoss()

    def preprocess_data_(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Preprocess a training/validation batch before forwarding through the model.

        The default implementation simply unpacks the batch into inputs and targets. Subclasses
        can override this method to perform normalization, augmentation or other processing.

        Args:
            batch (Tensor): A batch yielded by the dataloader, typically a tuple of
                (inputs, targets) or similar.

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing (inputs, targets).
        """
        x, y = batch
        return x, y

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the autoencoder model.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Reconstructed tensor produced by the decoder with the same shape as the
                input (B, C, H, W).
        """
        return self.model(x)

    def step(self, batch: Tensor, batch_idx: int, mode_prefix: str) -> Tensor:
        """Common step logic for training and validation.

        Unpacks and preprocesses the batch, runs a forward pass, computes reconstruction
        losses and logs several metrics to the internal average meters.

        Args:
            batch (Tensor): A batch from the dataloader (inputs, targets).
            batch_idx (int): Index of the current batch within the epoch.
            mode_prefix (str): Prefix used to separate train/val metrics (e.g. "train" or
                "val").

        Returns:
            Tensor: The primary loss tensor used for optimization.
        """
        x, y = self.preprocess_data_(batch)
        x_recon = self.forward(x)
        x_recon_bin = (x_recon > 0.5).float()
        loss = self.loss_f(x_recon, x)
        self.log_step(
            logs={"loss": loss, "MSE": F.mse_loss(x_recon, x), "MAE": F.l1_loss(x_recon, x),
                  "MSE_binary": F.mse_loss(x_recon_bin, x), "MAE_binary": F.l1_loss(x_recon_bin, x)},
            prefix=mode_prefix
        )

        return loss

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Training step called by Lightning during training.

        Args:
            batch (Tensor): A batch from the training dataloader.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Training loss tensor.
        """
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Validation step called by Lightning during validation.

        Args:
            batch (Tensor): A batch from the validation dataloader.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Validation loss tensor.
        """
        return self.step(batch, batch_idx, "val")

    def configure_model(self):
        """Create and return the autoencoder neural network.

        Returns:
            nn.Module: The autoencoder module (encoder + decoder) ready for forward passes.
        """
        return nn.Sequential(
            # Encoder
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),

            # # Bottleneck
            # nn.Flatten(),
            # nn.Linear(256 * 2 * 2, 256),
            # nn.ReLU(True),
            # nn.Linear(256, 256 * 2 * 2),
            # nn.ReLU(True),
            # nn.Unflatten(1, (256, 2, 2)),

            # Decoder
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def configure_optimizers(self) -> Optimizer:
        """
        Configure (create instance) the optimizer.
        
        Returns:
            Optimizer: The optimizer instance to be used for training, configured according to the parameters specified in the configuration dictionary.
        """
        opt_conf = self.conf['optimizer']
        return torch.optim.Adam(self.parameters(),
                                lr=opt_conf['lr'],
                                betas=(opt_conf['beta_1'], opt_conf['beta_2']),
                                weight_decay=opt_conf['weight_decay'])
