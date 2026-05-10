import math
import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class StraightLine(Dataset):

    def __init__(self,
                 img_h: Optional[int] = 32,
                 img_w: Optional[int] = 32,
                 num_images: Optional[int] = 50,
                 n_black_pixels: Optional[int] = 1,
                 transform: Optional[Callable] = None):
        """Dataset that generates synthetic images containing a single straight line.

        Args:
            img_h (Optional[int]): Image height in pixels. Defaults to 32.
            img_w (Optional[int]): Image width in pixels. Defaults to 32.
            num_images (Optional[int]): Number of samples in the dataset. Defaults to 50.
            n_black_pixels (Optional[int]): Number of black (missing) pixels to insert
                on the line to simulate discontinuities. Defaults to 1.
            transform (Optional[Callable]): Optional transform applied to PIL image
                before returning (e.g. torchvision transforms). If None, ToTensor
                is applied by default.
        """
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.num_images = num_images
        self.transform = transform
        self.n_black_pixels = n_black_pixels

        if self.transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def __len__(self):
        """Return the number of images in the dataset.

        Returns:
            int: The configured number of images (num_images).
        """
        return self.num_images

    def _get_random_line_coords(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Generate coordinates for a random straight line within the image bounds.

        The method either creates a mostly horizontal or mostly vertical line by
        choosing endpoints on opposite edges with a small margin.

        Returns:
            Tuple[Tuple[int,int], Tuple[int,int]]: Two (x, y) endpoint coordinates
                specifying the line within image dimensions.
        """
        if random.random() < 0.5:
            x1 = 2
            x2 = self.img_w - 2
            y1 = random.randint(2, self.img_h - 2)
            y2 = self.img_h - y1
        else:
            y1 = 2
            y2 = self.img_h - 2
            x1 = random.randint(2, self.img_w - 2)
            x2 = self.img_w - x1

        return (x1, y1), (x2, y2)

    def _create_l_image(self, line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Image.Image:
        """Create a grayscale PIL image with a white straight line drawn on black.

        Args:
            line_coords (Optional[Tuple[Tuple[int,int], Tuple[int,int]]]): Pair of
                (x, y) coordinates for the line endpoints. If None an empty image
                is created (caller typically supplies coordinates).

        Returns:
            PIL.Image: Grayscale (mode 'L') image containing the drawn line.
        """
        img = Image.new('L', (self.img_w, self.img_h), color=0)
        draw = ImageDraw.Draw(img)
        draw.line(line_coords, fill=255, width=1)
        return img

    def _create_image(
            self,
            line_coords: Tuple[Tuple[int, int], Tuple[int, int]],
            n_black_pixels: Optional[int] = None
    ):
        """Create an image with a white line and optional black gaps at the center.

        The image is generated in grayscale, the specified line is drawn in white
        and, if n_black_pixels > 0, a contiguous block of black pixels is inserted
        around the midpoint of the line to create a discontinuity.

        Args:
            line_coords (Tuple[Tuple[int,int], Tuple[int,int]]): Endpoints of the line.
            n_black_pixels (Optional[int]): Number of black pixels to set on the line's
                center. If None, uses the dataset default. Defaults to None.

        Returns:
            PIL.Image or Tensor: Transformed image. If a transform is configured the
                returned object may be a tensor (e.g. from torchvision.transforms).
        """
        if n_black_pixels is None:
            n_black_pixels = self.n_black_pixels

        img = self._create_l_image(line_coords)

        # add a black pixel in the middle (discontinuous line)
        if n_black_pixels > 0:
            img = np.array(img)
            line_center = (line_coords[0][0] + line_coords[1][0]) // 2, (line_coords[0][1] + line_coords[1][1]) // 2
            all_line_coords = np.argwhere(img > 128)
            center_point_idx = np.sum(np.abs(
                all_line_coords - np.array([line_center[1], line_center[0]]).reshape(1, 2).repeat(
                    all_line_coords.shape[0], axis=0)), axis=1).argmin()
            n_black = min(n_black_pixels, all_line_coords.shape[0] - 2)
            lower_idx = center_point_idx - n_black // 2
            upper_idx = center_point_idx + (n_black - (center_point_idx - lower_idx))
            idxs = np.array([list(all_line_coords[i]) for i in range(lower_idx, upper_idx)])
            img[idxs[:, 0], idxs[:, 1]] = 0
            img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            img = self.transform(img)

        return img

    def get_item(
            self,
            idx: int,
            line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
            n_black_pixels: Optional[int] = 0,
    ):
        """Return an image sample and metadata for a given index.

        Args:
            idx (int): Index requested by caller (not used to generate content).
            line_coords (Optional[Tuple[Tuple[int,int], Tuple[int,int]]]): If
                provided, these coordinates are used for the line; otherwise a
                random line is generated.
            n_black_pixels (Optional[int]): Number of black pixels to insert
                at the line center. Defaults to 0.

        Returns:
            tuple: (image, metadata) where image is the transformed image (PIL or
                tensor) and metadata is a dict containing 'line_coords' and
                'angle' (float, radians).
        """
        if line_coords is None:
            line_coords = self._get_random_line_coords()

        images = self._create_image(line_coords, n_black_pixels=n_black_pixels)
        return images, {'line_coords': line_coords, 'angle': math.atan(
            (line_coords[1][1] - line_coords[0][1]) / (1e-10 + line_coords[1][0] - line_coords[0][0]))}

    def __getitem__(self, idx: int):
        return self.get_item(idx)
