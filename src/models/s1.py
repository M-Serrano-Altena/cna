from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Fabric
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torchvision import utils


class Conv2dFixedFilters(nn.Module):
    """
    Fixed 2D convolutional layer with 4 filters that detect straight lines.
    """

    def __init__(self, fabric: Fabric, use_larger_weights: Optional[bool] = False, threshold_f: Optional[str] = "None"):
        """
        Initialize a Conv2dFixedFilters layer with predefined kernels for line detection.
        
        This layer contains 4 fixed convolutional filters designed to detect straight lines
        in four different orientations (vertical, diagonal-right, horizontal, diagonal-left).
        The filters use either smaller weights (sum to 1.0) or larger weights (sum to 3.0)
        depending on the use_larger_weights parameter.
        
        Args:
            fabric (Fabric): Lightning Fabric instance for device management.
            use_larger_weights (Optional[bool]): If True, uses larger filter weights that sum to 3.0.
                Defaults to False, which uses smaller weights summing to 1.0.
            threshold_f (Optional[str]): Activation function type to apply after convolution.
                Can be "None" (ReLU), "threshold" (binary thresholding), or "bernoulli".
                Defaults to "None".
        """
        super(Conv2dFixedFilters, self).__init__()
        self.threshold_f = threshold_f

        if use_larger_weights:
            # These kernels can sum up to 3.0 (with the proper 3 cells being active)
            self.weight = torch.tensor([
                [[[+0.0, -0.5, +0.0, -0.5, +0.0],
                  [+0.0, -0.5, +1.0, -0.5, +0.0],
                  [+0.0, -0.5, +1.0, -0.5, +0.0],
                  [+0.0, -0.5, +1.0, -0.5, +0.0],
                  [+0.0, -0.5, +0.0, -0.5, +0.0]]],
                [[[+0.0, +0.0, -0.5, -0.5, +0.0],
                  [+0.0, -0.5, +0.0, +1.0, -0.5],
                  [-0.5, +0.0, +1.0, +0.0, -0.5],
                  [-0.5, +1.0, +0.0, -0.5, +0.0],
                  [+0.0, -0.5, -0.5, +0.0, +0.0]]],
                [[[+0.0, +0.0, +0.0, +0.0, +0.0],
                  [-0.5, -0.5, -0.5, -0.5, -0.5],
                  [+0.0, +1.0, +1.0, +1.0, +0.0],
                  [-0.5, -0.5, -0.5, -0.5, -0.5],
                  [+0.0, +0.0, +0.0, +0.0, +0.0]]],
                [[[+0.0, -0.5, -0.5, +0.0, +0.0],
                  [-0.5, +1.0, +0.0, -0.5, +0.0],
                  [-0.5, +0.0, +1.0, +0.0, -0.5],
                  [+0.0, -0.5, +0.0, +1.0, -0.5],
                  [+0.0, +0.0, -0.5, -0.5, +0.0]]]
                # Filter could be further improved by setting 4x +0 in the middle to -0.5
            ], dtype=torch.float32, requires_grad=False).to(fabric.device)

            self.weight = torch.tensor([
                [[[-0.5, -0.5, +0.0, -0.5, -0.5],
                  [+0.0, -0.5, +1.0, -0.5, +0.0],
                  [-0.5, -0.5, +2.0, -0.5, -0.5],
                  [+0.0, -0.5, +1.0, -0.5, +0.0],
                  [-0.5, -0.5, +0.0, -0.5, -0.5]]],
                [[[-0.5, +0.0, -0.5, -0.5, +0.0],
                  [+0.0, -0.5, -0.5, +1.0, -0.5],
                  [-0.5, -0.5, +2.0, -0.5, -0.5],
                  [-0.5, +1.0, -0.5, -0.5, +0.0],
                  [+0.0, -0.5, -0.5, +0.0, -0.5]]],
                [[[-0.5, +0.0, -0.5, +0.0, -0.5],
                  [-0.5, -0.5, -0.5, -0.5, -0.5],
                  [+0.0, +1.0, +2.0, +1.0, +0.0],
                  [-0.5, -0.5, -0.5, -0.5, -0.5],
                  [-0.5, +0.0, -0.5, +0.0, -0.5]]],
                [[[+0.0, -0.5, -0.5, +0.0, -0.5],
                  [-0.5, +1.0, -0.5, -0.5, +0.0],
                  [-0.5, -0.5, +2.0, -0.5, -0.5],
                  [+0.0, -0.5, -0.5, +1.0, -0.5],
                  [-0.5, +0.0, -0.5, -0.5, +0.0]]]
                # Filter could be further improved by setting 4x +0 in the middle to -0.5
            ], dtype=torch.float32, requires_grad=False).to(fabric.device)

        else:
            # These kernels can sum up to 1.0 (with the proper 3 cells being active)
            self.weight = torch.tensor([[[[+0, -1, +0, -1, +0],
                                          [+0, -1, +2, -1, +0],
                                          [+0, -1, +2, -1, +0],
                                          [+0, -1, +2, -1, +0],
                                          [+0, -1, +0, -1, +0]]],
                                        [[[+0, +0, -1, -1, +0],
                                          [+0, -1, +0, +2, -1],
                                          [-1, +0, +2, +0, -1],
                                          [-1, +2, +0, -1, +0],
                                          [+0, -1, -1, +0, +0]]],
                                        [[[+0, +0, +0, +0, +0],
                                          [-1, -1, -1, -1, -1],
                                          [+0, +2, +2, +2, +0],
                                          [-1, -1, -1, -1, -1],
                                          [+0, +0, +0, +0, +0]]],
                                        [[[+0, -1, -1, +0, +0],
                                          [-1, +2, +0, -1, +0],
                                          [-1, +0, +2, +0, -1],
                                          [+0, -1, +0, +2, -1],
                                          [+0, +0, -1, -1, +0]]],
                                        # Filter could be further improved by setting 4x +0 in the middle to -1
                                        ], dtype=torch.float32, requires_grad=False).to(fabric.device)
            self.weight = self.weight / 6

    def apply_conv(self, x: Tensor) -> Tensor:
        """
        Apply 2D convolution with fixed line-detection filters to the input.
        
        Performs a padded convolution to maintain spatial dimensions, applying all 4 fixed
        filters to detect lines in different orientations. The output preserves the input
        spatial resolution through same-padding.
        
        Args:
            x (Tensor): Input image tensor of shape (batch, channels, height, width).
        
        Returns:
            Tensor: Convolved feature maps of shape (batch, 4, height, width).
        """
        x = F.conv2d(x, self.weight, padding="same")
        return x

    def apply_activation(self, a: Tensor) -> Tensor:
        """
        Apply the configured activation function to convolved features.
        
        Applies one of three activation strategies: ReLU (keep positive values),
        binary thresholding (convert to 0 or 1), or stochastic Bernoulli sampling.
        The specific function is determined by the threshold_f parameter set during
        initialization.
        
        Args:
            a (Tensor): Feature tensor from convolution of shape (batch, filters, height, width).
        
        Returns:
            Tensor: Activated features of the same shape as input.
        """

        if self.threshold_f == "None":
            return torch.where(a > 0, a, 0.)
        elif self.threshold_f == "threshold":
            return torch.where(a > 0, 1., 0.)
        elif self.threshold_f == "bernoulli":
            return torch.bernoulli(torch.clip(a, 0, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Execute the forward pass through the fixed-filter feature extractor.
        
        Handles both 4D input (batch, channels, height, width) and 5D input
        (batch, time_steps, channels, height, width) by applying convolution and
        activation to each frame independently. 5D inputs are processed sequentially
        and then recombined along the time dimension.
        
        Args:
            x (Tensor): Input tensor, either 4D or 5D shape.
        
        Returns:
            Tensor: Extracted and activated features with added filter dimension.
        """
        if len(x.shape) == 5:
            result = []
            for idx in range(x.shape[1]):
                result.append(self.apply_conv(x[:, idx, ...]))
            a = torch.stack(result, dim=1)
        else:
            a = self.apply_conv(x).unsqueeze(1)
        return self.apply_activation(a)


class FixedFilterFeatureExtractor(pl.LightningModule):
    """
    PyTorch Lightning module that uses a CNN with a fixed filter.
    """

    def __init__(self, conf: Dict[str, Any], fabric: Fabric) -> None:
        """
        Initialize the FixedFilterFeatureExtractor Lightning module.
        
        Sets up a Lightning module that wraps the Conv2dFixedFilters layer for
        feature extraction using predefined fixed convolution kernels. This module
        integrates with PyTorch Lightning training pipelines.
        
        Args:
            conf (Dict[str, Any]): Configuration dictionary containing model parameters,
                including nested 'feature_extractor' and 's1_params' keys.
            fabric (Fabric): Lightning Fabric instance for distributed training and device management.
        """
        super().__init__()
        self.conf = conf
        self.fabric = fabric
        self.model = self.configure_model()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the fixed-filter feature extraction model.
        
        Delegates to the underlying Conv2dFixedFilters model to extract line-detection
        features from the input image using the fixed convolutional kernels.
        
        Args:
            x (Tensor): Input image tensor.
        
        Returns:
            Tensor: Extracted feature maps from line-detection filters.
        """
        return self.model(x)

    def configure_model(self) -> nn.Module:
        """
        Create and configure the Conv2dFixedFilters model instance.
        
        Instantiates the fixed-filter convolutional layer using parameters from
        the configuration dictionary, specifically from conf['feature_extractor']['s1_params'].
        
        Returns:
            nn.Module: Configured Conv2dFixedFilters instance ready for feature extraction.
        """
        return Conv2dFixedFilters(self.fabric, **self.conf['feature_extractor']['s1_params'])

    def plot_model_weights(self, show_plot: Optional[bool] = False) -> List[Path]:
        """
        Generate and save visualizations of the model filter weights.
        
        Creates comprehensive weight visualizations including a histogram showing
        the distribution of weight values and a grid visualization of all filter kernels.
        Plots are saved to disk if a store path is configured, and optionally displayed.
        
        Args:
            show_plot (Optional[bool]): If True, display the plots using matplotlib.
                Defaults to False.
        
        Returns:
            List[Path]: List of file paths where the generated plots were saved.
                Returns empty list if no store path is configured.
        """

        def _hist_plot(ax, weight, title):
            bins = 20
            min, max = torch.min(weight).item(), torch.max(weight).item()
            hist = torch.histc(weight, bins=bins, min=min, max=max)
            x = np.linspace(min, max, bins)
            ax.bar(x, hist, align='center')
            ax.set_xlabel(f'Bins form {min:.4f} to {max:.4f}')
            ax.set_title(title)

        def _plot_weights(fig, ax, weight, title):
            weight_img_list = [weight[i, j].unsqueeze(0) for j in range(weight.shape[1]) for i in
                               range(weight.shape[0])]
            # Order is [(0, 0), (1, 0), ..., (3, 0), (0, 1), ..., (3, 7)]
            # The columns show the output channels, the rows the input channels
            grid = utils.make_grid(weight_img_list, nrow=weight.shape[0], normalize=True, scale_each=True, pad_value=1)
            # grid = grid / 2 - 1/6  # Normalize to [-1/6, 1/3]
            im = ax.imshow(grid[:, 2:-2, 2:-2].permute(1, 2, 0), interpolation='none',
                           cmap="gray")  # , vmin=-1/6, vmax=1/3)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        files = []
        for layer, weight in [('feature extractor', self.model.weight)]:
            fig, axs = plt.subplots(1, 2, figsize=(16, 10))
            _hist_plot(axs[0], weight.detach().cpu(), f"Weight distribution ({layer})")
            _plot_weights(fig, axs[1], weight[:20, :20, ...].detach().cpu(), f"Weight matrix ({layer})")
            plt.tight_layout()

            fig_fp = self.conf['run']['plots'].get('store_path', None)

            if fig_fp is not None and fig_fp != "None":
                fp = Path(fig_fp) / f'weights_{layer}.png'
                plt.savefig(fp)
                files.append(fp)

            if show_plot:
                plt.show()

            plt.close()
        return files

