from typing import Union
from numbers import Number
import numpy as np
import torch

numeric = Union[int, float, complex, np.number, torch.Tensor]


class AverageMeter:
    """Track a running weighted mean for streaming numeric values.

    This utility is useful for aggregating metrics across multiple updates,
    such as averaging loss values over mini-batches with different batch sizes.
    The internal state is defined by the cumulative weight (`count`) and the
    current weighted mean (`mean`).
    """

    def __init__(self) -> None:
        """Initialize an empty meter.

        The meter starts with zero accumulated weight and a mean of zero,
        and can then be updated incrementally using the method `add` or
        the method `__call__`.

        Returns:
            None
        """
        self.count = 0
        self.mean = 0

    def reset(self) -> None:
        """Reset the meter to its initial empty state.

        After calling this method, both the accumulated weight (`count`) and
        running mean (`mean`) are set back to zero.

        Returns:
            None
        """
        self.count = 0
        self.mean = 0

    def __call__(self, value: numeric, weight: numeric = 1) -> None:
        """Add a new weighted observation using call syntax.

        This is a convenience wrapper around :meth:`add`, allowing the meter
        instance to be used like a function.

        Args:
            value (numeric): Observation to incorporate into the running mean.
            weight (numeric): Weight associated with ``value`` (for example,
                batch size when averaging batch-level metrics). Defaults to 1.

        Returns:
            None
        """
        self.add(value, weight)

    def add(self, value: numeric, weight: numeric = 1) -> None:
        """Update the running weighted mean with a new observation.

        The mean is updated incrementally using the previous cumulative weight
        and mean, so no history of past values needs to be stored.

        Args:
            value (numeric): Observation to add to the meter.
            weight (numeric): Weight for ``value``. In training loops, this is
                commonly the batch size. Defaults to 1.

        Returns:
            None
        """
        self.mean = (self.mean * self.count + value * weight) / (self.count + weight)
        self.count += weight