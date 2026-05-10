from typing import Dict, List, Optional, Union

from lightning.fabric.loggers.logger import Logger
from lightning.pytorch.loggers import WandbLogger

from utils.custom_print import print_logs


class ConsoleLogger:
    """
    Simple console logger that formats and prints metric dictionaries.

    This logger groups metrics by prefix (the part before a "/" in the
    metric key) and prints each group using the project's custom
    print_logs helper. If a metric key does not contain a prefix, it is
    placed under the "general" group along with the provided step.
    """

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int]) -> None:
        """
        Format and print metrics to the console.

        Groups metrics by prefix when a metric key contains a '/'. For
        example, a key 'train/loss' will be grouped under 'train' with
        the subkey 'loss'. Keys without a '/' are collected under the
        'general' group. The optional step is included in the 'general'
        group as 'step'. Each group is printed using print_logs.

        Args:
            metrics (Dict[str, float]): Mapping of metric names to values.
            step (Optional[int]): Current global step or iteration.

        Returns:
            None: This method only prints to the console and does not
            return a value.
        """
        logs_by_prefix: Dict[str, Dict[str, Union[int, float, None]]] = {"general": {'step': step}}
        for k, v in metrics.items():
            if "/" in k:
                prefix, metric = k.split("/", 1)
                if prefix not in logs_by_prefix:
                    logs_by_prefix[prefix] = {}
                logs_by_prefix[prefix][metric] = v
            else:
                logs_by_prefix["general"][k] = v

        for k, v in logs_by_prefix.items():
            print_logs(v, title=k)


def loggers_from_conf(conf: Dict) -> List[Logger]:
    """
    Build logger instances from a configuration dictionary.

    The configuration is expected to contain a 'logging' section where
    each entry specifies a logger name and its settings. Inactive
    loggers (logger_conf['active'] is False) are skipped. Supported
    logger names (case-insensitive) include:
      - 'wandb': Instantiates a lightning.pytorch WandbLogger with the
        provided project, save_dir, log_model, job_type and group.
      - 'console': Instantiates the local ConsoleLogger.

    Args:
        conf (Dict): Configuration dictionary containing a 'logging'
            mapping with logger names and their respective settings.

    Returns:
        List[Logger]: A list of instantiated logger objects compatible
        with Lightning Fabric/PyTorch Lightning logging interfaces.

    Raises:
        NotImplementedError: If a logger name is present in the
            configuration but is not supported by this factory.
    """
    loggers = []
    for logger_name, logger_conf in conf['logging'].items():
        if not logger_conf['active']:
            continue
        if logger_name.lower() == "wandb":
            loggers.append(WandbLogger(project=logger_conf['project'],
                                       save_dir=logger_conf['save_dir'],
                                       log_model=logger_conf['log_model'],
                                       config=conf,
                                       job_type=logger_conf['job_type'],
                                       group=logger_conf['group']))
        elif logger_name.lower() == "console":
            loggers.append(ConsoleLogger())
        else:
            raise NotImplementedError(f"Logger {logger_name} not implemented.")
    return loggers