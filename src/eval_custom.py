"""
eval_custom.py — Run CNA or autoencoder on a custom PNG image.

Usage (CNA):
    python eval_custom.py net-fragments \
        --load ../checkpoints/net-fragments.ckpt \
        --custom_image ../NPS_64.png \
        --act_threshold 0.5 --square_factor 0.6 0.8 1.0 1.2 1.4 1.6

Usage (Autoencoder):
    python eval_custom.py autoencoder \
        --load ../checkpoints/autoencoder.ckpt \
        --custom_image ../NPS_64.png
"""

import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# ── Patch get_data_generator and parse_args BEFORE main_evaluation runs ──────

import main_evaluation as _me

_original_parse_args = _me.parse_args
_original_get_data_generator = _me.get_data_generator


def _patched_parse_args(parser=None):
    parser = _original_parse_args(parser)
    parser.add_argument(
        "--custom_image",
        type=str,
        default=None,
        help="Path to a custom PNG image (grayscale). Overrides --line_type.",
    )
    # Allow --line_type to default to 'custom' without crashing
    for action in parser._actions:
        if hasattr(action, 'dest') and action.dest == 'line_type':
            action.choices = ['straight', 'curved', 'digits', 'objects', 'custom']
            action.default = 'custom'
    return parser


def _patched_get_data_generator(config):
    custom_path = config.get('custom_image', None)
    if custom_path is None:
        return _original_get_data_generator(config)

    # Load the PNG, convert to float tensor [1, H, W]
    img = Image.open(custom_path).convert('L')
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]

    n_samples = config.get('n_samples', 20)
    print(f"  Custom image loaded: {custom_path}  shape={tuple(tensor.shape)}")
    print(f"  Repeating {n_samples} times for the generator.")

    # Each item is a tuple (image_tensor,) — same format as the regular datasets
    return [(tensor,)] * n_samples


_me.parse_args = _patched_parse_args
_me.get_data_generator = _patched_get_data_generator

# ── Also patch configure so custom_image ends up in config ────────────────────

from main_training import configure as _original_configure


def _patched_configure(parser=None):
    config = _original_configure(parser)
    # Pull --custom_image out of sys.argv manually if present
    if '--custom_image' in sys.argv:
        idx = sys.argv.index('--custom_image')
        config['custom_image'] = sys.argv[idx + 1]
        # Set line_type to 'custom' so the video path uses a nice folder name
        config['line_type'] = 'custom'
    return config


import main_training as _mt
_mt.configure = _patched_configure
_me.configure = _patched_configure  # make sure main_evaluation uses the patched version too


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _me.main()
