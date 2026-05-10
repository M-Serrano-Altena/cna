import random
from typing import Callable, Optional

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

class SplineLine(Dataset):
    """Dataset that procedurally generates binary images containing one spline-like line.

    Each sample is a single-channel image where line pixels are set to 1 and all
    remaining pixels are 0.
    """

    def __init__(self,
                 img_h: Optional[int] = 32,
                 img_w: Optional[int] = 32,
                 num_images: Optional[int] = 50,
                 n_spline_points: Optional[int] = 2,
                 transform: Optional[Callable] = None):
        """Initialize the spline-line dataset.

        Args:
            img_h (Optional[int]): Height of each generated image in pixels.
            img_w (Optional[int]): Width of each generated image in pixels.
            num_images (Optional[int]): Total number of samples to generate.
            n_spline_points (Optional[int]): Number of interior control points used
                to define the interpolated line.
            transform (Optional[Callable]): Transform applied to each generated
                image. If ``None``, a default ``ToTensor`` transform is used.
        """
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.num_images = num_images
        self.n_spline_points = n_spline_points
        self.transform = transform

        if self.transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

        self.lines = [self._get_random_line() for _ in range(num_images)]

    def __len__(self):
        """Return the number of generated samples.

        Returns:
            int: Dataset size.
        """
        return self.num_images

    def _get_random_line(self) -> np.ndarray:
        """Generate one binary image containing a random interpolated line.

        The line is built by interpolating random y-values over evenly spaced
        x-control points, then filling vertical gaps to keep connectivity between
        adjacent x positions. With 50% probability, the image is transposed.

        Returns:
            np.ndarray: A 2D NumPy array of shape ``(img_h, img_w)`` with values in
            ``{0, 1}``.
        """

        x = np.linspace(2.0, self.img_w - 2, num=self.n_spline_points+2)
        y = np.random.randint(2, self.img_h - 2, size=self.n_spline_points+2)
        x2 = np.linspace(0, self.img_w-1, num=self.img_w)
        y = np.interp(x2, x, y).round()
        x = x2.round()

        diff = (y[:31] - y[1:])
        line_holes = (np.abs(diff) > 1) * (np.abs(diff) - 1)
        missing_x, missing_y = [], []
        for idx in np.where(line_holes)[0]:
            holes = line_holes[idx]
            val = y[idx]
            for h in range(holes.astype(int)):
                val -= np.sign(diff[idx])
                missing_x.append(idx)
                missing_y.append(val)

        y = np.append(y, missing_y).astype(int)
        x = np.append(x, missing_x).astype(int)

        img = np.zeros((self.img_h, self.img_w))
        img[y, x] = 1

        if random.random() < 0.5:
            # rotate the line
            img = img.T

        return img

    def __getitem__(self, idx: int):
        """Return a transformed sample and an empty target dictionary.

        Args:
            idx (int): Sample index.

        Returns:
            tuple[torch.Tensor, dict]: A tuple containing the transformed image
            tensor (``float32``) and an empty metadata/target dictionary.
        """
        return self.transform(self.lines[idx]).type(torch.float32), {}


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = SplineLine()
    for i in range(10):
        img = dataset[i]
        plt.imshow(img[0].cpu().numpy(), cmap='gray', vmin=0, vmax=1, interpolation='none')
        plt.tight_layout()
        plt.show()