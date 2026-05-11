import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import lightning.pytorch as pl
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image, ImageDraw, ImageFont
from lightning import Fabric
from skimage.draw import circle_perimeter, line
from torch import Tensor
from tqdm import tqdm

from data.spline_line import SplineLine
from data.straight_line import StraightLine
from main_autoencoder import setup_autoencoder
from main_training import configure, setup_fabric, setup_feature_extractor, setup_lateral_network, setup_wandb
from utils.custom_print import print_start
from utils.meters import AverageMeter
from utils.store_load_run import load_run

RESULTS_BASE_DIR = Path("results")


def parse_args(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Create and parse command line arguments for model evaluation.

    This function builds an ArgumentParser (unless one is provided), registers
    all command-line options used by the evaluation script and returns the
    populated parser. Options include settings for number of samples to
    evaluate, noise level, line interruption length, line type, evaluation
    framerate, and paths for storing or loading baseline activations. Several
    options also map directly to nested configuration keys used elsewhere in
    the project (for example `lateral_model:s2_params:act_threshold`).

    Args:
        parser: An optional existing argparse.ArgumentParser instance. If
            provided, the arguments will be added to it; otherwise a new
            parser is created.

    Returns:
        argparse.ArgumentParser: The argument parser with evaluation options
            configured. The caller can call `parse_args()` on the returned
            object to obtain the parsed values.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Model Evaluation")

    parser.add_argument("--n_samples",
                        type=int,
                        metavar="N",
                        default=59,
                        help="Number of samples to evaluate."
                        )
    parser.add_argument("--noise",
                        type=float,
                        metavar="N",
                        default=0.0,
                        help="Ratio of noise to add."
                        )
    parser.add_argument("--line_interrupt",
                        type=int,
                        metavar="N",
                        default=0,
                        help="Number of pixels to remove from the line."
                        )
    parser.add_argument("--line_type",
                        choices=['straight', 'curved', 'digits', 'objects'],
                        type=str,
                        default='straight',
                        )
    parser.add_argument("--fps",
                        type=int,
                        metavar="N",
                        default=10,
                        help="Number of samples to evaluate."
                        )
    parser.add_argument("--store_baseline_activations_path",
                        type=str,
                        default=None,
                        help="Store baseline activations to compare models to."
                        )
    parser.add_argument("--load_baseline_activations_path",
                        type=str,
                        default=None,
                        help="Load baseline activations to compare models to."
                        )
    parser.add_argument("--wandb_type",
                        type=str,
                        default="eval",
                        dest='logging:wandb:job_type',
                        )
    parser.add_argument("--act_threshold",
                        default="bernoulli",
                        dest='lateral_model:s2_params:act_threshold',
                        )
    parser.add_argument("--square_factor",
                        type=float,
                        nargs='+',
                        default=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                        dest='lateral_model:s2_params:square_factor',
                        )

    return parser


def load_models() -> Tuple[Dict[str, Any], Fabric, pl.LightningModule, pl.LightningModule]:
    """
    Load configuration, initialize Fabric, and prepare models for evaluation.

    This function parses command-line arguments, creates the training/evaluation
    Fabric, and instantiates the feature extractor and either a lateral
    network or an autoencoder depending on the configuration. It then loads
    model weights from the configured checkpoint, sets the models to
    evaluation mode, and returns the loaded configuration, Fabric instance,
    the feature extractor, and the main model.

    Returns:
        tuple: A tuple containing (config, fabric, feature_extractor, model).
    """
    config = configure(parse_args())
    fabric = setup_fabric(config)
    feature_extractor, _ = setup_feature_extractor(config, fabric)
    if 'lateral_model' in config:
        model = setup_lateral_network(config, fabric)
    else:
        model, _ = setup_autoencoder(config, fabric)

    assert 'load_state_path' in config['run'] and config['run']['load_state_path'] != 'None', \
        "Please specify a path to a trained model."

    config, state = load_run(config, fabric)
    feature_extractor.load_state_dict(state['feature_extractor'])
    model.load_state_dict(state['lateral_network'])

    feature_extractor.eval()
    model.eval()

    return config, fabric, feature_extractor, model


def get_datapoints(n_points) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    def rotate_point_square(p: Tuple[int, int]) -> Tuple[int, int]:
        """
        Rotate a 2D integer grid point counter-clockwise along a square path.

        The rotation follows a fixed square trajectory used to generate pairs
        of endpoints for the straight-line dataset. Given the current
        coordinate, the function computes the next coordinate along the
        perimeter of a predefined square region.

        Args:
            p: Current (x, y) integer coordinate.

        Returns:
            Tuple[int, int]: Next (x, y) coordinate after one CCW step.
        """
        (x, y) = p
        if x == 2:
            if y < 30:
                y += 1
            else:
                x += 1
        elif x < 30:
            if y == 2:
                x -= 1
            else:
                x += 1
        else:
            if y > 2:
                y -= 1
            else:
                x -= 1
        return (x, y)

    p1 = (2, 16)
    p2 = (30, 16)
    points = [(p1, p2)]

    for idx in range(n_points - 1):
        p1 = rotate_point_square(p1)
        p2 = rotate_point_square(p2)
        points.append((p1, p2))
    return points


class CustomImage:
    """
    Helper to compose visualization images for network activations.

    This class builds a reusable template with static labels and layout and
    provides methods to convert internal activation tensors into colored
    masks and heatmaps, then assemble them together with the input image to
    produce a single visual frame representing the model's internal state.
    """

    FONTS_DIR = Path("fonts")

    def __init__(self):
        """
        Initialize the class.
        """
        self.img_size = 256
        self.img_template = self.create_template_image()

    def to_mask(self, mask: np.array) -> np.array:
        """
        Convert a multi-channel binary mask to an RGB color image.

        Each channel in the input mask is assigned a color from a colormap
        and blended into a 3-channel (H, W, 3) image. The input is expected
        in shape (C, H, W) with values in {0,1} or [0,1].

        Args:
            mask: Numpy array of shape (C, H, W) representing channel masks.

        Returns:
            np.array: RGB image of shape (H, W, 3) in uint8.
        """

        mask_colors = matplotlib.colormaps['gist_rainbow'](range(0, 256, 256 // mask.shape[0]))
        result = np.zeros((3, mask.shape[1], mask.shape[2]))
        for channel in range(mask.shape[0]):
            mask_c = np.ones_like(result) * (mask_colors[channel, :3] * 255).astype(int).reshape(3, 1, 1)
            mask_idx = np.repeat((mask[channel] > 0.5)[np.newaxis, :, :], 3, axis=0)
            result[mask_idx] = np.clip(result[mask_idx] + mask_c[mask_idx], a_min=0, a_max=255)
        return result.astype("uint8").transpose(1, 2, 0)

    def to_heatmap(self, activation_probabilities: np.array) -> np.array:
        """
        Convert activation probability maps to an RGB heatmap image.

        The function collapses across channels by taking the per-pixel
        maximum probability and maps the resulting single-channel image to a
        colored heatmap using OpenCV's COLORMAP_HOT, then converts to RGB.

        Args:
            activation_probabilities: Numpy array of shape (C, H, W) with
                probability values in [0, 1].

        Returns:
            np.array: RGB heatmap image of shape (H, W, 3) in uint8.
        """
        heatmap = (np.max(activation_probabilities, axis=0) * 255).astype("uint8")
        heatmap = cv2.cvtColor(cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)
        return heatmap

    def create_template_image(self) -> Image:
        """
        Create the static layout image used as a base for visualizations.

        This method composes the overall canvas including title text,
        section labels, logo, decorative borders and arrow indicators. The
        resulting PIL Image is used as a template onto which dynamic tiles
        (input, feature masks, activations, heatmaps) are pasted for each
        visualization frame.

        Returns:
            PIL.Image: Template RGB image sized to accommodate tiles and
                annotations.
        """
        outer_padding = 30
        inner_padding = 80
        inner_dist = 80
        font_size_padded = 30
        font_size = 20
        line_padding = 2

        title_size = int(font_size * 1.7)
        self.height = 2 * outer_padding + inner_padding + 2 * font_size_padded + 1 * self.img_size + title_size
        self.width = 2 * outer_padding + 3 * inner_dist + 4 * self.img_size
        output = Image.new("RGB", (self.width, self.height), (255, 255, 255))

        # Paste Images
        self.h_center, self.w1 = self.height // 2 - (self.img_size + font_size_padded) // 2 + title_size, outer_padding
        self.w2 = self.w1 + self.img_size + inner_dist  # S1
        self.w3 = self.w2 + self.img_size + inner_dist  # S2
        self.w4 = self.w3 + self.img_size + inner_dist  # S2 Probabilities

        logo = Image.open(self.FONTS_DIR / "ZHAW.png").resize((60, 60), Image.Resampling.LANCZOS)
        output.paste(logo, (20, self.height - 20 - 60), mask=logo)

        # Add Texts
        font_title = ImageFont.truetype(self.FONTS_DIR / "calibrib.ttf", title_size)
        font = ImageFont.truetype(self.FONTS_DIR / "calibrib.ttf", font_size)
        font_foot = ImageFont.truetype(self.FONTS_DIR / "calibri_italic.ttf", int(font_size * 0.8))
        draw = ImageDraw.Draw(output)
        draw.text((outer_padding, outer_padding),
                  "Dynamic Link Architecture (DNA)", (0, 100, 166),
                  font=font_title)
        draw.text((self.w1, self.h_center - font_size_padded), "Input Image", (40, 40, 40), font=font)
        draw.text((self.w2, self.h_center - font_size_padded), "Feature Activation (S1)", (40, 40, 40), font=font)
        draw.text((self.w3, self.h_center - font_size_padded), "Net Fragments (S2)", (40, 40, 40), font=font)
        draw.text((self.w4, self.h_center - font_size_padded), "Net Fragments (S2) Probabilities", (40, 40, 40),
                  font=font)

        draw.text((20 + 60 + 10, self.height - 20 - font_size_padded), "Sager et al.", (128, 128, 128),
                  font=font_foot)

        # Draw rectangle
        draw.rounded_rectangle((self.w2 - 20, self.h_center - font_size_padded - 10, self.width - 15, self.height - 80),
                               outline="#0064a6",
                               width=3,
                               radius=7)

        # Draw arrows using cv2
        output = np.array(output)
        output = cv2.arrowedLine(output, (self.w1 + self.img_size + line_padding, self.h_center + self.img_size // 2),
                                 (self.w2 - line_padding, self.h_center + self.img_size // 2), (128, 128, 128), 5)
        output = cv2.arrowedLine(output, (self.w2 + self.img_size + line_padding, self.h_center + self.img_size // 2),
                                 (self.w3 - line_padding, self.h_center + self.img_size // 2),
                                 (128, 128, 128), 5)

        return Image.fromarray(output)

    def create_image(self, img: Tensor, s1_in_features: Tensor, s2_act: Tensor, s2_act_prob: Tensor) -> np.array:
        """
        Assemble a visualization frame for a single timestep.

        Converts tensors for the input image, S1 feature masks, discrete S2
        activations and S2 activation probabilities into PIL images, resizes
        them to the configured tile size and pastes them onto the template
        to produce a final RGB frame suitable for writing to a video.

        Args:
            img: Input image tensor with values in [0,1].
            s1_in_features: Binary feature maps for S1 (tensor).
            s2_act: Binary S2 activations (tensor).
            s2_act_prob: S2 activation probability maps (tensor).

        Returns:
            np.array: Final RGB frame as a NumPy array (H, W, 3).
        """
        assert 0. <= img.min() and img.max() <= 1., "img must be in [0, 1]"
        # assert 0. <= s1_in_features.min() and s1_in_features.max() <= 1., "in_features must be in [0, 1]"
        assert 0. <= s2_act.min() and s2_act.max() <= 1., "s2_act must be in [0, 1]"
        assert 0. <= s2_act_prob.min() and s2_act_prob.max() <= 1., "s2_act_prob must be in [0, 1]"

        s1_in_features = torch.where(s1_in_features > 0., 1., 0.)

        img = Image.fromarray((img * 255).squeeze().cpu().numpy().astype("uint8")).convert("RGB")
        s1_in_features = Image.fromarray(self.to_mask((s1_in_features * 255).squeeze().cpu().numpy()))
        s2_act = Image.fromarray(self.to_mask((s2_act * 255).squeeze().cpu().numpy()))
        s2_act_prob = Image.fromarray(self.to_heatmap((s2_act_prob).squeeze().cpu().numpy()))

        # Resize Images
        img = img.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
        s1_in_features = s1_in_features.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
        s2_act = s2_act.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
        s2_act_prob = s2_act_prob.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)

        output = self.img_template.copy()

        # Paste Images
        output.paste(img, (self.w1, self.h_center))
        output.paste(s1_in_features, (self.w2, self.h_center))
        output.paste(s2_act, (self.w3, self.h_center))
        output.paste(s2_act_prob, (self.w4, self.h_center))

        # output.show()
        return np.array(output)


def get_data_generator(config: Dict[str, Any]):
    """
    Yield input samples and metadata according to the configured dataset type.

    Depending on config['line_type'], this factory yields different kinds of
    synthetic images: straight moving lines, curved splines, digit strokes,
    or simple objects. Each yielded item is a tuple (image_tensor, meta).

    Args:
        config (Dict[str, Any]): Configuration dictionary containing dataset parameters such
            as 'line_type', 'n_samples' and 'line_interrupt'.

    Yields:
        Tuple[Tensor, Dict]: A pair of the input image tensor and an
            associated metadata dictionary for each sample.
    """
    if config['line_type'] == 'straight':
        points = get_datapoints(config['n_samples'])
        dataset = StraightLine(num_images=len(points))
        for i in range(len(points)):
            img, meta = dataset.get_item(i, line_coords=points[i], n_black_pixels=config['line_interrupt'])
            yield img, meta

    elif config['line_type'] == 'curved':
        dataset = SplineLine(num_images=config['n_samples'])
        for i in range(config['n_samples']):
            img, meta = dataset[i]
            yield img, meta

    elif config['line_type'] == 'digits':
        scale = 2

        def draw_digit(img, digit):
            segments = {
                '0': [(0, 0, 10, 0), (0, 0, 0, 18), (10, 0, 10, 18), (0, 18, 10, 18)],
                '1': [(0, 0, 0, 18)],
                '2': [(0, 0, 10, 0), (10, 0, 10, 9), (0, 9, 10, 9), (0, 9, 0, 18), (0, 18, 10, 18)],
                '3': [(0, 0, 10, 0), (10, 0, 10, 18), (0, 9, 10, 9), (0, 18, 10, 18)],
                '4': [(0, 0, 0, 9), (0, 9, 10, 9), (10, 0, 10, 18)],
                '5': [(10, 0, 0, 0), (0, 0, 0, 9), (0, 9, 10, 9), (10, 9, 10, 18), (0, 18, 10, 18)],
                '6': [(10, 0, 0, 0), (0, 0, 0, 18), (0, 9, 10, 9), (10, 9, 10, 18), (0, 18, 10, 18)],
                '7': [(0, 0, 10, 0), (10, 0, 10, 18)],
                '8': [(0, 0, 10, 0), (0, 0, 0, 18), (10, 0, 10, 18), (0, 9, 10, 9), (0, 18, 10, 18)],
                '9': [(0, 18, 10, 18), (10, 0, 10, 18), (0, 9, 10, 9), (0, 0, 0, 9), (0, 0, 10, 0)],
            }

            x_coords = [x * scale for line in segments[digit] for x in (line[0], line[2])]
            y_coords = [y * scale for line in segments[digit] for y in (line[1], line[3])]

            # Find min and max values
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            offset_y = int(max(0, 32 - ((max_y - min_y) // 2)))
            offset_x = int(max(0, 32 - ((max_x - min_x) // 2)))

            for segment in segments[digit]:
                rr, cc = line(scale * segment[1] + offset_y, scale * segment[0] + offset_x,
                              scale * segment[3] + offset_y, scale * segment[2] + offset_x)
                img[rr, cc] = 1

        for i, digit in enumerate('1234567890'):
            img = np.zeros((64, 64), dtype=np.uint8)
            draw_digit(img, digit)
            yield torch.from_numpy(img).unsqueeze(0).float().to('cuda'), {}

    elif config['line_type'] == 'objects':
        scale = 3

        objects = {
            'triangle': [('line', 0, 10, 5, 0), ('line', 5, 0, 10, 10), ('line', 10, 10, 0, 10)],
            'circle': [('circle', 5, 5, 5)],
            'house': [('line', 0, 10, 0, 5), ('line', 0, 5, 5, 1), ('line', 5, 1, 10, 5), ('line', 10, 5, 10, 10),
                      ('line', 0, 10, 10, 10)],
            'star': [('line', 5, 0, 6, 4), ('line', 6, 4, 10, 4), ('line', 10, 4, 7, 6), ('line', 7, 6, 8, 10),
                     ('line', 8, 10, 5, 7),
                     ('line', 5, 7, 2, 10), ('line', 2, 10, 3, 6), ('line', 3, 6, 0, 4), ('line', 0, 4, 4, 4),
                     ('line', 4, 4, 5, 0)],
            'letter_A': [('line', 0, 10, 5, 0), ('line', 5, 0, 10, 10), ('line', 2, 6, 8, 6)],
            'arrow': [('line', 2, 0, 2, 10), ('line', 2, 0, 0, 3), ('line', 2, 0, 4, 3)],
            'smile': [('line', 0, 4, 6, 4), ('line', 0, 4, 1, 6), ('line', 1, 6, 5, 6), ('line', 5, 6, 6, 4),
                      ('line', 1, 0, 1, 1), ('line', 5, 0, 5, 1)],
            'sun': [('line', 5, 0, 5, 2), ('line', 5, 8, 5, 10), ('line', 0, 5, 2, 5), ('line', 8, 5, 10, 5),
                    ('line', 2, 2, 8, 8),
                    ('line', 2, 8, 8, 2)],
            'umbrella': [('line', 4, 0, 0, 5), ('line', 4, 0, 8, 5), ('line', 0, 5, 8, 5), ('line', 4, 5, 4, 10),
                          ('line', 3, 10, 4, 10)],
            'clock2': [('circle', 4, 4, 4), ('line', 4, 4, 4, 1), ('line', 4, 4, 6, 4)],
            'submarine': [('line', 3, 0, 3, 2), ('line', 3, 2, 1, 3), ('line', 1, 3, 0, 5), ('line', 0, 5, 1, 7),
                          ('line', 1, 7, 3, 8), ('line', 3, 8, 7, 8), ('line', 7, 8, 9, 5), ('line', 9, 5, 7, 2),
                          ('line', 7, 2, 3, 2), ('line', 7, 4, 7, 6)],
            'face': [('circle', 5, 5, 5), ('circle', 3, 4, 2), ('circle', 7, 4, 2), ('line', 3, 7, 7, 7)],
            'flower': [('circle', 4, 4, 3),
                       ('circle', 1, 1, 1),
                       ('circle', 7, 1, 1),
                       ('circle', 1, 7, 1),
                       ('circle', 7, 7, 1)],

            'wheel': [('circle', 4, 4, 4),
                      ('circle', 4, 4, 2),
                      ('line', 4, 0, 4, 8),
                      ('line', 0, 4, 8, 4),
                      ('line', 1, 1, 7, 7),
                      ('line', 1, 7, 7, 1)],
            'target': [('circle', 6, 6, 6),
                       ('circle', 6, 6, 4),
                       ('circle', 6, 6, 2)],
            'sun_with_rays': [('circle', 5, 5, 3),
                              ('line', 5, 0, 5, 2),
                              ('line', 5, 8, 5, 10),
                              ('line', 0, 5, 2, 5),
                              ('line', 8, 5, 10, 5),
                              ('line', 1, 1, 3, 3),
                              ('line', 7, 7, 9, 9),
                              ('line', 1, 9, 3, 7),
                              ('line', 7, 3, 9, 1)],
            'magnifying_glass': [('circle', 4, 4, 3),
                                 ('line', 0, 0, 3, 3)],
            'planet_with_moons': [('circle', 4, 4, 4),
                                  ('circle', 7, 4, 2),
                                  ('circle', 1, 5, 1)],
        }

        def draw_object(img, object_name):
            for scale in [6, 5, 4, 3, 2, 1]:
                scale_ok = True
                img2 = img.copy()

                line_segments = []
                circle_segments = []
                y_max, x_max = 0, 0
                y_min, x_min = 64, 64

                for segment in objects[object_name]:
                    if segment[0] == 'line':
                        line_segments.append([int(round(s * scale, 0)) for s in segment[1:]])
                        y_max = max(y_max, max(line_segments[-1][1], line_segments[-1][3]))
                        x_max = max(x_max, max(line_segments[-1][0], line_segments[-1][2]))
                        y_min = min(y_min, min(line_segments[-1][1], line_segments[-1][3]))
                        x_min = min(x_min, min(line_segments[-1][0], line_segments[-1][2]))
                    elif segment[0] == 'circle':
                        circle_segments.append([int(round(s * scale, 0)) for s in segment[1:]])
                        y_max = max(y_max, circle_segments[-1][1] + circle_segments[-1][2])
                        x_max = max(x_max, circle_segments[-1][0] + circle_segments[-1][2])
                        y_min = min(y_min, circle_segments[-1][1] - circle_segments[-1][2])
                        x_min = min(x_min, circle_segments[-1][0] - circle_segments[-1][2])

                offset_y = int(max(0, 32 - ((y_max - y_min) // 2)))
                offset_x = int(max(0, 32 - ((x_max - x_min) // 2)))
                for segment in line_segments:
                    try:
                        rr, cc = line(segment[1] + offset_y, segment[0] + offset_x, segment[3] + offset_y,
                                      segment[2] + offset_x)
                        img2[rr, cc] = 1
                    except IndexError:
                        scale_ok = False
                        break

                for segment in circle_segments:
                    try:
                        rr, cc = circle_perimeter(segment[1] + offset_y, segment[0] + offset_x, segment[2])
                        img2[rr, cc] = 1
                    except IndexError:
                        scale_ok = False
                        break

                if not scale_ok:
                    img2 = img.copy()
                    continue
                else:
                    break

            return img2

        object_names = list(objects.keys())
        for i, object_name in enumerate(object_names):
            img = np.zeros((64, 64), dtype=np.uint8)
            img = draw_object(img, object_name)
            yield torch.from_numpy(img).unsqueeze(0).float().to('cuda'), {}

    else:
        raise ValueError(f"Unknown line type: {config['line_type']}")


def merge_alt_channels(config: Dict[str, Optional[Any]], lateral_features: List[Tensor]) -> List[Tensor]:
    """
    Collapse alternative-channel representations into original channel axes.

    Some lateral representations use an "alternative cells" strategy where
    each logical channel is represented by multiple alternative channels
    (only one active at a time). This function reshapes and takes the
    element-wise maximum across alternatives to restore the original channel
    dimensionality for visualization and analysis.

    Args:
        config (Dict[str, Optional[Any]]): Configuration dict containing 'n_alternative_cells' and
            lateral_model channel count.
        lateral_features (List[Tensor]): List of tensors representing lateral activations
            at different timesteps.

    Returns:
        List[Tensor]: Lateral features with alternatives merged into the
            original channel dimension.
    """
    n_alt = config['n_alternative_cells']
    n_channels = config['lateral_model']['channels']

    result = []
    for lf in lateral_features:
        lf = lf.reshape(-1, n_channels, n_alt, lf.shape[2], lf.shape[3])
        assert torch.sum((lf > 0), dim=2).max() <= 1, "Only one channel per alternative channel can be active"
        result.append(torch.max(lf, dim=2)[0])

    return result


def analyze_noise(noise: Tensor, random_mask: Tensor, lateral_features: List[Tensor]) -> float:
    """
    Compute the fraction of noisy elements that were corrected by the model.

    Given a binary mask of positions that were flipped (random_mask) and the
    ground-truth noise values, this function inspects the final lateral
    activations to determine how many noisy positions were changed back to
    their original values. The metric is returned as a float ratio in [0,1].

    Args:
        noise (Tensor): Tensor containing the original noise values for masked
            positions.
        random_mask (Tensor): 1D boolean tensor indexing the flattened positions that
            were flipped.
        lateral_features (List[Tensor]): List of lateral activation tensors; the final
            timestep is used for the evaluation.

    Returns:
        float: Fraction of noisy positions that were corrected.
    """
    lateral_features = lateral_features[-1].view(-1)
    lateral_features_noise = lateral_features[random_mask]
    removed_noise_ratio = torch.sum(lateral_features_noise != noise) / torch.numel(noise)
    return removed_noise_ratio.item()


def analyze_recon_error(lateral_features: List[Tensor], baseline_lateral_features: List[Tensor]) -> Tuple[
    float, float, float]:
    """
    Compute reconstruction accuracy, recall and precision between two sets of
    lateral activations.

    The function compares the final-timestep flattened lateral features to a
    baseline activation vector using L1 loss transformed into a similarity
    score (1 - L1). It returns three metrics: overall accuracy, recall on
    baseline-active positions, and precision on predicted-active positions.

    Args:
        lateral_features (List[Tensor]): List of tensors of predicted lateral activations
            across timesteps.
        baseline_lateral_features (List[Tensor]): List of tensors containing baseline
            activations for comparison.

    Returns:
        Tuple[float, float, float]: (accuracy, recall, precision) as floats.
    """
    lateral_features = lateral_features[-1].view(-1)
    baseline_lateral_features = baseline_lateral_features[-1].view(-1)
    accuracy = 1. - F.l1_loss(lateral_features, baseline_lateral_features)
    recall = 1. - F.l1_loss(lateral_features[baseline_lateral_features > 0.],
                            baseline_lateral_features[baseline_lateral_features > 0.])
    precision = 1. - F.l1_loss(lateral_features[lateral_features > 0.],
                               baseline_lateral_features[lateral_features > 0.])
    return accuracy.item(), recall.item(), precision.item()


def analyze_interrupt_line(img: Tensor,
                           baseline_img: Tensor,
                           lateral_features: List[Tensor],
                           baseline_lateral_features: List[Tensor]) -> float:
    """
    Measure reconstruction accuracy specifically over interrupted pixels.

    This function identifies pixels that differ between the current input
    image and a baseline (uninterrupted) image, extracts the corresponding
    S2 activations from predicted and baseline lateral features, and
    computes an L1-based accuracy score over only those positions that were
    active in the baseline. If no baseline-active positions exist in the
    masked region, a default score of 0.0 is returned.

    Args:
        img (Tensor): Current input image tensor.
        baseline_img (Tensor): Baseline (reference) image tensor.
        lateral_features (List[Tensor]): Predicted lateral features (list of tensors).
        baseline_lateral_features (List[Tensor]): Baseline lateral features (list of tensors).

    Returns:
        float: Reconstruction accuracy in [0,1] over interrupted pixels.
    """
    baseline_img = baseline_img[-1].squeeze()
    mask = torch.where(img != baseline_img, True, False).unsqueeze(0).repeat(lateral_features[-1].shape[1], 1, 1)
    baseline_lateral_features_masked = baseline_lateral_features[-1].squeeze(0)[mask]
    lateral_features_masked = lateral_features[-1].squeeze(0)[mask]
    mask_active = torch.where(baseline_lateral_features_masked > 0., True, False)
    if baseline_lateral_features_masked[mask_active].numel() > 0:
        accuracy = 1. - F.l1_loss(lateral_features_masked[mask_active], baseline_lateral_features_masked[mask_active])
    else:
        accuracy = torch.tensor(0.)
    # mask = torch.where(img != baseline_img, True, False)
    # baseline_lateral_features_masked = torch.clip(torch.sum(baseline_lateral_features[-1].squeeze(0), dim=0)[mask],
    # max=1)
    # lateral_features_masked = torch.clip(torch.sum(lateral_features[-1].squeeze(0), dim=0)[mask], max=1)
    # accuracy = 1. - F.l1_loss(lateral_features_masked, baseline_lateral_features_masked)
    return accuracy.item()


def step_lateral_model(
        config: Dict[str, Any],
        batch: Tensor,
        features: Tensor,
        model: pl.LightningModule
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
    """
    Run the lateral (S2) recurrent model across configured timesteps.

    For each timestep the model is advanced, the combined input (sensory
    features and current S2 state) is passed through the module to produce
    continuous and binary S2 activations. The function collects and
    returns lists of inputs, input features, discrete lateral activations
    and continuous lateral activations for all timesteps.

    Args:
        config (Dict[str, Any]): Configuration dict containing lateral model parameters such
            as 'max_timesteps'.
        batch (Tensor): Raw input image tensor for the current sample.
        features (Tensor): Precomputed sensory features from the feature extractor.
        model (pl.LightningModule): The LightningModule implementing the lateral dynamics.

    Returns:
        Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]: Lists
            of (batch inputs, input features, binary lateral activations,
            continuous lateral activations) across timesteps.
    """
    input, input_features, lateral_features, lateral_features_float = [], [], [], []

    z = torch.zeros((features.shape[0], model.model.out_channels, features.shape[2],
                     features.shape[3]), device=batch.device)

    for t in range(config["lateral_model"]["max_timesteps"]):
        model.model.update_ts(t)
        x_in = torch.cat([features, z], dim=1)
        z_float, z = model(x_in)

        input.append(batch)
        input_features.append(features)
        lateral_features.append(z)
        lateral_features_float.append(z_float)

    return input, input_features, lateral_features, lateral_features_float

def step_autoencoder(batch: Tensor, features: Tensor, model: pl.LightningModule) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
    """
    Single-step autoencoder forward to obtain binary and continuous codes.

    The autoencoder path is non-recurrent: the model computes a continuous
    code (clamped to [0,1]) which is then thresholded at 0.5 to obtain a
    binary activation map. The resulting lists mirror the timestep-based
    outputs from the recurrent lateral model for compatibility.

    Args:
        batch: Raw input tensor for the sample.
        features: Sensory features from the feature extractor.
        model: Autoencoder LightningModule.

    Returns:
        Tuple of lists: ([batch], [features], [binary_code], [continuous_code]).
    """
    z_float = model(features)
    z_float = torch.clamp(z_float, min=0., max=1.)
    z = torch.where(z_float > 0.5, 1., 0.)

    return [batch], [features], [z], [z_float]


def predict_sample(
        config: Dict[str, Optional[Any]],
        fabric: Fabric,
        feature_extractor: pl.LightningModule,
        model: pl.LightningModule,
        batch: Tensor,
        batch_idx: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, float, float, Tuple[float, float, float]]:
    """
    Run feature extraction and lateral model inference for a single sample.

    The function moves the batch to the Fabric device, computes sensory
    features, optionally injects synthetic noise, and then runs either the
    recurrent lateral model or the autoencoder path. It also computes
    analysis metrics such as removed noise ratio and reconstruction
    measures by comparing to stored baseline activations if available.

    Args:
        config (Dict[str, Optional[Any]]): Configuration dictionary controlling model behavior.
        fabric (Fabric): Fabric instance providing device placement.
        feature_extractor (pl.LightningModule): Feature extractor LightningModule.
        model (pl.LightningModule): Lateral network or autoencoder LightningModule.
        batch (Tensor): Single-sample batch tensor produced by the data generator.
        batch_idx (int): Integer index of the current sample.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, float, float, Tuple[float, float, float]]: Stacked tensors for inputs, input features, binary lateral
            activations, continuous lateral activations, removed_noise
            ratio, interrupt_line_reconstruction score, and reconstruction
            error tuple.
    """


    with torch.no_grad():
        batch = batch[0].to(fabric.device)
        features = feature_extractor(batch.unsqueeze(0)).squeeze(1)

        if config['noise'] > 0.:
            features_s = features.shape
            num_elements = features.numel()
            num_flips = int(config['noise'] * num_elements)
            random_mask = torch.randperm(num_elements)[:num_flips]
            random_mask = torch.zeros(num_elements, dtype=torch.bool).scatter(0, random_mask, 1)
            features = features.view(-1)
            noise = 1.0 - features[random_mask]
            features[random_mask] = noise
            features = features.view(features_s)

        if 'lateral_model' in config:
            input, input_features, lateral_features, lateral_features_float = step_lateral_model(config, batch, features, model)
            lateral_features = merge_alt_channels(config, lateral_features)
            lateral_features_float = merge_alt_channels(config, lateral_features_float)
        else:
            input, input_features, lateral_features, lateral_features_float = step_autoencoder(batch, features, model)

    removed_noise = analyze_noise(noise, random_mask, lateral_features) if config['noise'] > 0. else 0
    if 'load_baseline_activations_path' in config and config['load_baseline_activations_path'] is not None:
        t = torch.load(config['load_baseline_activations_path'])
        recon_error = analyze_recon_error(lateral_features, t[0][batch_idx])
        interrupt_line_recon = analyze_interrupt_line(batch[0], t[1][batch_idx], lateral_features, t[0][batch_idx])
    else:
        recon_error = (-1, -1, -1)
        interrupt_line_recon = -1
    return (torch.stack(input), torch.stack(input_features), torch.stack(lateral_features),
            torch.stack(lateral_features_float), removed_noise, interrupt_line_recon, recon_error)


def process_data(
        generator: Any,
        config: Dict[str, Any],
        fabric: Fabric,
        feature_extractor: pl.LightningModule,
        model: pl.LightningModule,
) -> Tuple[float, float, float, float, float, Dict[str, Any]]:
    """
    Iterate over data, run model inference, collect metrics and save video.

    For each sample yielded by the generator this function runs prediction,
    constructs visualization frames, writes them into an MP4 file, and
    aggregates metrics such as noise reduction and reconstruction scores.
    Optionally stores baseline activations and logs summary statistics
    suitable for wandb reporting.

    Args:
        generator (Any): Iterable yielding (image, meta) tuples.
        config (Dict[str, Any]): Configuration dict controlling dataset and model options.
        fabric (Fabric): Fabric instance for device placement.
        feature_extractor (pl.LightningModule): Feature extractor LightningModule.
        model (pl.LightningModule): Lateral network or autoencoder LightningModule.

    Returns:
        Tuple[float, float, float, float, float, Dict[str, Any]]: Aggregated metrics (noise_reduction, line_recon_acc,
            recon_accuracy, recon_recall, recon_precision, logs dict).
    """

    logs = {}
    imgs_, s2_acts = [], []
    ci = CustomImage()
    avg_noise_meter = AverageMeter()
    avg_line_recon_accuracy_meter = AverageMeter()
    avg_recon_accuracy_meter, avg_recon_recall_meter, avg_recon_precision_meter = (AverageMeter(), AverageMeter(),
                                                                                   AverageMeter())
    if 'lateral_model' in config:
        fp = (
            f"{RESULTS_BASE_DIR}/{config['line_type']}/{config['config']}/th-"
            f"{str(config['lateral_model']['s2_params']['act_threshold'])}_sf-"
            f"{config['lateral_model']['s2_params']['square_factor'][0]}-"
            f"{config['lateral_model']['s2_params']['square_factor'][-1]}/"
            f"{'noise-' + str(config['noise']) if config['noise'] > 0 else 'no-noise'}_li-"
            f"{config['line_interrupt']}.mp4")
    else:
        fp = (
            f"{RESULTS_BASE_DIR}/{config['line_type']}/{config['config']}/"
            f"{'noise-' + str(config['noise']) if config['noise'] > 0 else 'no-noise'}_li-"
            f"{config['line_interrupt']}.mp4")

    if not Path(fp).parent.exists():
        Path(fp).parent.mkdir(parents=True)

    if Path(fp).exists():
        Path(fp).unlink()
    print(fp)
    out = cv2.VideoWriter(fp, cv2.VideoWriter_fourcc(*'mp4v'), config['fps'],
                          (ci.width, ci.height))
    for i, img in tqdm(enumerate(generator), total=config["n_samples"]):
        inp, s1_inp_features, s2_act, s2_act_prob, removed_noise, interrupt_line_recon, recon_error = predict_sample(
            config, fabric, feature_extractor, model, img, i)
        s2_act_prob = torch.where((s2_act > 0.) | (s1_inp_features > 0.), s2_act_prob, torch.zeros_like(s2_act_prob))
        s2_acts.append(s2_act)
        imgs_.append(inp)
        avg_noise_meter(removed_noise)
        avg_line_recon_accuracy_meter(interrupt_line_recon)
        avg_recon_accuracy_meter(recon_error[0])
        avg_recon_recall_meter(recon_error[1])
        avg_recon_precision_meter(recon_error[2])

        for timestep in range(s2_act.shape[0]):
            result = ci.create_image(inp[timestep], s1_inp_features[timestep, 0], s2_act[timestep, 0],
                                     s2_act_prob[timestep, 0])
            out.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    out.release()
    logs['act_video'] = wandb.Video(fp, fps=config['fps'], format="mp4")

    s2_acts = torch.stack(s2_acts)
    if 'store_baseline_activations_path' in config and config['store_baseline_activations_path'] is not None:
        torch.save([s2_acts, torch.stack(imgs_)], config['store_baseline_activations_path'])

    calc_overlap = True
    if calc_overlap:
        # consider final timestep
        s2_acts_f = s2_acts[:, -1].squeeze()

        # calculate active / inactive cells
        avg_activate_cells = torch.mean(torch.sum(s2_acts_f > 0, dim=(1, 2, 3)).float())
        avg_inactivate_cells = torch.mean(torch.sum(s2_acts_f == 0, dim=(1, 2, 3)).float())
        print(f"Average activated cells in net fragments: {avg_activate_cells}")
        print(f"Average inactivated cells in net fragments: {avg_inactivate_cells}")

        # calculate overlap
        s2_acts_f = s2_acts_f.view(s2_acts_f.shape[0], -1)
        overlaps, boverlaps = [], []
        for v in range(s2_acts_f.shape[0]):
            biggest_overlap = 0
            for v2 in range(s2_acts_f.shape[0]):
                if v != v2:
                    overlap = torch.sum(s2_acts_f[v] * s2_acts_f[v2]) / torch.sum(s2_acts_f[v] + s2_acts_f[v2])
                    overlaps.append(overlap)
                    if overlap > biggest_overlap:
                        biggest_overlap = overlap
            boverlaps.append(biggest_overlap)

        overlaps = torch.tensor(overlaps).float()
        boverlaps = torch.tensor(boverlaps).float()
        print(f"Average overlap: {torch.mean(overlaps)}")
        print(f"Average biggest overlap: {torch.mean(boverlaps)}")

        logs['avg_activate_cells'] = avg_activate_cells
        logs['avg_inactivate_cells'] = avg_inactivate_cells
        logs['avg_overlap'] = torch.mean(overlaps)
        logs['avg_biggest_overlap'] = torch.mean(boverlaps)

    print(f"Average Noise Reduction: {avg_noise_meter.mean}")
    print(f"Average Interrupt Line Reconstruction Accuracy: {avg_line_recon_accuracy_meter.mean}")
    print(f"Average Reconstruction Accuracy: {avg_recon_accuracy_meter.mean}")
    print(f"Average Reconstruction Recall: {avg_recon_recall_meter.mean}")
    print(f"Average Reconstruction Precision: {avg_recon_precision_meter.mean}")

    logs['avg_noise_reduction'] = avg_noise_meter.mean
    logs['avg_line_recon_accuracy'] = avg_line_recon_accuracy_meter.mean
    logs['avg_recon_accuracy'] = avg_recon_accuracy_meter.mean
    logs['avg_recon_recall'] = avg_recon_recall_meter.mean
    logs['avg_recon_precision'] = avg_recon_precision_meter.mean

    return (avg_noise_meter.mean, avg_line_recon_accuracy_meter.mean, avg_recon_accuracy_meter.mean,
            avg_recon_recall_meter.mean, avg_recon_precision_meter.mean, logs)


def store_experiment_results(noise_reduction: float,
                             avg_line_recon_accuracy_meter: float,
                             recon_accuracy: float,
                             recon_recall: float,
                             recon_precision: float,
                             config: Dict[str, Any]) -> None:
    """
    Append summary experiment metrics and configuration to a JSONL file.

    The function writes a JSON object with the provided metrics and the
    full configuration to a per-experiment results file inside RESULTS_BASE_DIR.
    The parent directory is created if missing. Each call appends one JSON
    line to the file, enabling accumulation of multiple runs.

    Args:
        noise_reduction (float): Mean fraction of noise removed by the model.
        avg_line_recon_accuracy_meter (float): Mean reconstruction accuracy on
            interrupted line segments.
        recon_accuracy (float): Overall reconstruction accuracy metric.
        recon_recall (float): Reconstruction recall metric.
        recon_precision (float): Reconstruction precision metric.
        config (Dict[str, Any]): Configuration dictionary used to run the experiment.
    """
    fp = f"{RESULTS_BASE_DIR}/{config['config']}/experiment_results.json"
    print(fp)
    # Ensure parent directory exists before attempting to write
    Path(fp).parent.mkdir(parents=True, exist_ok=True)

    with open(fp, "a") as f:
        json.dump({'config': config, 'noise_reduction': noise_reduction,
                   'avg_line_recon_accuracy_meter': avg_line_recon_accuracy_meter, 'recon_accuracy': recon_accuracy,
                   'recon_recall': recon_recall, 'recon_precision': recon_precision}, f)
        f.write("\n")


def main() -> None:
    """
    Entry point for evaluation: load models, run data processing and log results.

    This function orchestrates the full evaluation pipeline: it loads the
    trained models, sets up wandb logging, iterates over the dataset to
    create visualization videos and metrics, stores aggregate results and
    finalizes logging if enabled.
    """
    print_start("Starting python script 'main_evaluation.py'...",
                title="Evaluating Model and Print activations")
    config, fabric, feature_extractor, model = load_models()
    setup_wandb(config)
    generator = get_data_generator(config)
    noise_reduction, avg_line_recon_accuracy_meter, recon_accuracy, recon_recall, recon_precision, logs = process_data(
        generator, config, fabric, feature_extractor, model)
    if 'store_baseline_activations_path' not in config or config['store_baseline_activations_path'] is None:
        store_experiment_results(noise_reduction, avg_line_recon_accuracy_meter, recon_accuracy, recon_recall,
                                 recon_precision, config)

    if "wandb" in config['logging'].keys() and config['logging']['wandb']['active']:
        wandb.log(logs)
        wandb.finish()


if __name__ == "__main__":
    main()
