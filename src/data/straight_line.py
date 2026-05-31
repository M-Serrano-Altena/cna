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


class UniformSlopeStraightLine(StraightLine):
    """Straight-line dataset with one precomputed slope per sample.

    Instead of sampling line endpoints randomly for every item, this variant
    generates a fixed set of line orientations up front, one per requested
    image, and shuffles the resulting samples. This makes the orientation
    distribution much closer to uniform for small datasets.
    """

    def __init__(
        self,
        img_h: Optional[int] = 32,
        img_w: Optional[int] = 32,
        num_images: Optional[int] = 50,
        n_black_pixels: Optional[int] = 1,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            img_h=img_h,
            img_w=img_w,
            num_images=num_images,
            n_black_pixels=n_black_pixels,
            transform=transform,
        )
        self._uniform_line_coords = self._generate_uniform_line_coords()
        random.shuffle(self._uniform_line_coords)

    def _line_coords_from_angle(self, angle: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Create a line segment through the image center for a given angle."""
        assert self.img_w is not None and self.img_h is not None
        img_w = self.img_w
        img_h = self.img_h
        cx = (img_w - 1) / 2.0
        cy = (img_h - 1) / 2.0
        dx = math.cos(angle)
        dy = math.sin(angle)
        eps = 1e-12
        points = []

        if abs(dx) > eps:
            for x in (0.0, float(img_w - 1)):
                y = cy + (x - cx) * (dy / dx)
                if 0.0 <= y <= float(img_h - 1):
                    points.append((x, y))

        if abs(dy) > eps:
            for y in (0.0, float(img_h - 1)):
                x = cx + (y - cy) * (dx / dy)
                if 0.0 <= x <= float(img_w - 1):
                    points.append((x, y))

        unique_points = []
        for point in points:
            rounded_point = (int(round(point[0])), int(round(point[1])))
            if rounded_point not in unique_points:
                unique_points.append(rounded_point)

        if len(unique_points) < 2:
            return (0, int(round(cy))), (img_w - 1, int(round(cy)))

        return unique_points[0], unique_points[1]

    def _generate_uniform_line_coords(self) -> list[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Generate one line segment per uniformly spaced orientation."""
        assert self.num_images is not None
        num_images = self.num_images

        if num_images <= 0:
            return []

        angles = np.linspace(0.0, math.pi, num=num_images, endpoint=False)
        return [self._line_coords_from_angle(float(angle)) for angle in angles]

    def get_item(
        self,
        idx: int,
        line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        n_black_pixels: Optional[int] = 0,
    ):
        if line_coords is None:
            if not self._uniform_line_coords:
                return super().get_item(idx, line_coords=None, n_black_pixels=n_black_pixels)
            line_coords = self._uniform_line_coords[idx % len(self._uniform_line_coords)]
        return super().get_item(idx, line_coords=line_coords, n_black_pixels=n_black_pixels)

