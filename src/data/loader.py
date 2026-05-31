import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from data.straight_line import StraightLine, UniformSlopeStraightLine
from torch.utils.data import DataLoader, Dataset

T = TypeVar('T')
_collate_fn_t = Callable[[List[T]], Any]
T_co = TypeVar('T_co', covariant=True)
_path_t = Union[str, os.PathLike, Path]


def _get_dataset(
        dataset_config: Dict,
) -> Tuple[Any, Optional[Any], Any]:
    """Build train/validation/test datasets from configuration.

    Args:
        dataset_config (Optional[Dict]): Dataset configuration dictionary
            containing ``train_dataset_params``, ``valid_dataset_params``, and
            ``test_dataset_params``.

    Returns:
        Tuple[Any, Optional[Any], Any]: Instantiated datasets in the order
        ``(train_set, valid_set, test_set)``.
    """
    params_keys = ["train_dataset_params", "valid_dataset_params", "test_dataset_params"]
    if not all(dataset_config.get(key) is not None for key in params_keys):
        raise ValueError("Missing dataset parameters in configuration.")
    
    uniform_sampling = dataset_config["train_dataset_params"].pop("uniform_sampling", False)
    dataset_config["valid_dataset_params"].pop("uniform_sampling", False)
    dataset_config["test_dataset_params"].pop("uniform_sampling", False)
    
    
    if uniform_sampling:
        train_set = UniformSlopeStraightLine(**dataset_config['train_dataset_params'])
    else:
        train_set = StraightLine(**dataset_config['train_dataset_params'])

    valid_set = StraightLine(**dataset_config['valid_dataset_params'])
    test_set = StraightLine(**dataset_config['test_dataset_params'])

    return train_set, valid_set, test_set


def _get_loader_safe(
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = True,
        collate_fn: Optional[_collate_fn_t] = None,
        shuffle: Optional[bool] = True,
        drop_last: Optional[bool] = False,
) -> Optional[DataLoader[T_co]]:
    """Create a ``DataLoader`` if a dataset is provided.

    Initializes deterministic seeding for the data loader generator and worker
    processes to improve reproducibility across runs.

    Args:
        dataset (Dataset[T_co]): Input dataset to wrap.
        batch_size (Optional[int]): Number of samples per batch.
        num_workers (Optional[int]): Number of worker subprocesses.
        pin_memory (Optional[bool]): Whether to pin memory for faster host to
            device transfers.
        collate_fn (Optional[_collate_fn_t]): Optional custom collation
            function.
        shuffle (Optional[bool]): Whether to shuffle dataset indices.
        drop_last (Optional[bool]): Whether to drop the last incomplete batch.

    Returns:
        Optional[DataLoader[T_co]]: Configured data loader, or ``None`` when
        ``dataset`` is ``None``.
    """
    if dataset is not None:
        train_gen = torch.Generator()
        train_gen.manual_seed(0)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
                          pin_memory=pin_memory, generator=train_gen, worker_init_fn=seed_worker, collate_fn=collate_fn)
    else:
        return None


def _get_torch_data_loaders(
        train_set: Optional[Dataset[T_co]] = None,
        valid_set: Optional[Dataset[T_co]] = None,
        test_set: Optional[Dataset[T_co]] = None,
        batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = True,
        collate_fn: Optional[_collate_fn_t] = None,
        shuffle_train: Optional[bool] = True,
        shuffle_valid: Optional[bool] = False,
        shuffle_test: Optional[bool] = False,
        drop_last_train: Optional[bool] = False,
        drop_last_valid: Optional[bool] = False,
        drop_last_test: Optional[bool] = False,
) -> Tuple[DataLoader[T_co], Optional[DataLoader[T_co]], Optional[DataLoader[T_co]]]:
    """Create loaders for train, validation, and test datasets.

    Args:
        train_set (Optional[Dataset[T_co]]): Training dataset.
        valid_set (Optional[Dataset[T_co]]): Validation dataset.
        test_set (Optional[Dataset[T_co]]): Test dataset.
        batch_size (Optional[int]): Number of samples per batch.
        num_workers (Optional[int]): Number of worker subprocesses.
        pin_memory (Optional[bool]): Whether to pin memory in each loader.
        collate_fn (Optional[_collate_fn_t]): Optional collate function shared
            across loaders.
        shuffle_train (Optional[bool]): Whether to shuffle the training set.
        shuffle_valid (Optional[bool]): Whether to shuffle the validation set.
        shuffle_test (Optional[bool]): Whether to shuffle the test set.
        drop_last_train (Optional[bool]): Whether to drop last incomplete batch
            in training.
        drop_last_valid (Optional[bool]): Whether to drop last incomplete batch
            in validation.
        drop_last_test (Optional[bool]): Whether to drop last incomplete batch
            in testing.

    Returns:
        Tuple[DataLoader[T_co], Optional[DataLoader[T_co]], Optional[DataLoader[T_co]]]:
        Data loaders in the order ``(train_loader, valid_loader, test_loader)``.
    """

    train_loader = _get_loader_safe(train_set, batch_size, num_workers, pin_memory, collate_fn, shuffle_train,
                                    drop_last_train)
    valid_loader = _get_loader_safe(valid_set, batch_size, num_workers, pin_memory, collate_fn, shuffle_valid,
                                    drop_last_valid)
    test_loader = _get_loader_safe(test_set, batch_size, num_workers, pin_memory, collate_fn, shuffle_test,
                                   drop_last_test)

    return train_loader, valid_loader, test_loader


def loaders_from_config(config: Dict) -> Union[Any, Any, Any]:
    """Build dataset loaders from a global configuration dictionary.

    Expects dataset settings under ``config["dataset"]`` and runtime loader
    settings under ``config["run"]``.

    Args:
        config (Dict): Full experiment or run configuration.

    Returns:
        Union[Any, Any, Any]: Tuple-like return containing train, validation,
        and test data loaders.
    """
    data_config = config["dataset"]
    train_set, valid_set, test_set = _get_dataset(data_config)
    return _get_torch_data_loaders(
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        batch_size=config["run"]["batch_size"],
        num_workers=config["run"]["num_workers"],
    )
