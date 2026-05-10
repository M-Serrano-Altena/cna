import operator
import os
from argparse import Namespace
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import yaml
from yaml.loader import SafeLoader

from utils.custom_print import print_exception, print_info_config, print_warn

_path_t = Union[str, os.PathLike, Path]

CONFIGS_DIR = Path("configs")
DATA_CONFIGS_FP = CONFIGS_DIR / "data.yaml"


def _load_config(path: _path_t) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the configuration file.

    Returns:
        A dictionary containing the loaded configuration.
    """
    with open(path) as f:
        return yaml.load(f, Loader=SafeLoader)


def get_from_nested_dict(data_dict: Dict, key_list: Union[List[str], str]):
    """Retrieve a value from a nested dictionary.

    Args:
        data_dict: Dictionary to retrieve the value from.
        key_list: List of keys defining the path to the value.
            A single string key is also accepted.

    Returns:
        The value stored at the specified nested key path.
    """
    if isinstance(key_list, str):
        key_list = [key_list]
    return reduce(operator.getitem, key_list, data_dict)


def set_in_nested_dict(data_dict: Dict, key_list: Union[List[str], str], value: Any):
    """Set a value in a nested dictionary.

    Args:
        data_dict: Dictionary in which to set the value.
        key_list: List of keys defining the path to the value.
            A single string key is also accepted.
        value: Value to set at the specified nested key path.
    """
    if isinstance(key_list, str):
        key_list = [key_list]
    get_from_nested_dict(data_dict, key_list[:-1])[key_list[-1]] = value


def _add_cli_args(config: Dict[str, Any], cli_args: Namespace) -> Dict[str, Any]:
    """Add command-line arguments to a configuration dictionary.

    Args:
        config: Configuration dictionary.
        cli_args: Parsed command-line arguments.

    Returns:
        The updated configuration dictionary with CLI arguments applied.
    """
    for key, value in vars(cli_args).items():
        if value is None:
            continue
        try:
            set_in_nested_dict(config, key.split(":"), value)
        except Exception as e:
            print_exception(e)
            print_warn(f"Could not set {key} to {value} in config.")
        if "cli_args" not in config:
            config["cli_args"] = {}
        config["cli_args"][key] = value
    return config


def get_config(
        config_name: str,
        cli_args: Optional[Namespace] = None,
) -> Dict[str, Any]:
    """Load and optionally modify a configuration.

    Args:
        config_name: Name of the configuration file without extension.
        cli_args: Optional parsed command-line arguments.

    Returns:
        A dictionary containing the final configuration.
    """
    config = _load_config(CONFIGS_DIR / f"{config_name}.yaml")
    if cli_args is not None:
        config = _add_cli_args(config, cli_args)
    print_info_config(config, "Config")
    return config
