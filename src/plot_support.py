from typing import List

import matplotlib.pyplot as plt
import numpy as np
import wandb


LAMBDA = 1.3 * 11 * 10000
X_MAX = 20

RUN_PATHS = {
    "net-fragments": "sagerpascal/net-fragments-final/m8ecvdhb",
}



def limit_support(x: List[float]) -> List[float]:
    """Clamp support values to the configured upper bound.

    This mirrors the training-time behavior where support values are limited
    before being stored or visualized.

    Args:
        x: Support values per cell.

    Returns:
        Support values with each entry capped at ``LAMBDA``.
    """
    return [min(x_i, LAMBDA) for x_i in x]


def get_support_from_wandb(run_id: str) -> tuple[List[float], List[float], List[float], List[float]]:
    """Fetch support metrics from a Weights & Biases run history.

    The function scans the run history, extracts the relevant support metrics,
    and clamps them to the same maximum value used during training.

    Args:
        run_id: Full W&B run path, for example ``"entity/project/run_id"``.

    Returns:
        A tuple containing the average active support, minimum active support,
        maximum active support, and average inactive support, in that order.
    """
    # Get the run's history
    api = wandb.Api()
    run = api.run(run_id)
    history = run.scan_history()

    # Get the support values
    avg_support_active = [h['S2/avg_support_active'] for h in history if h['S2/avg_support_active'] is not None]
    min_support_active = [h['S2/min_support_active'] for h in history if h['S2/min_support_active'] is not None]
    max_support_active = [h['S2/max_support_active'] for h in history if h['S2/max_support_active'] is not None]
    avg_support_inactive = [h['S2/avg_support_inactive'] for h in history if h['S2/avg_support_inactive'] is not None]

    # in the code, the support is limited to lambda -> consider this for this plot!
    avg_support_active = limit_support(avg_support_active)
    min_support_active = limit_support(min_support_active)
    max_support_active = limit_support(max_support_active)
    avg_support_inactive = limit_support(avg_support_inactive)

    return avg_support_active, min_support_active, max_support_active, avg_support_inactive

def print_support_active_cells(
    title: str,
    min_support_active: List[float],
    max_support_active: List[float],
    avg_support_active: List[float],
    avg_support_inactive: List[float],
) -> None:
    """Plot support values for active and inactive cells over time.

    The plot shows the mean support for active cells, the min/max range for
    active cells, and the mean support for inactive cells. It is intended for
    quick visual inspection of how support evolves across epochs.

    Args:
        title: Descriptive name of the run or experiment.
        min_support_active: Minimum support values for active cells.
        max_support_active: Maximum support values for active cells.
        avg_support_active: Average support values for active cells.
        avg_support_inactive: Average support values for inactive cells.

    Returns:
        None.
    """

    # if problems with font, run it on local machine
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Helvetica"
    x = np.arange(1, len(avg_support_active) + 1, 1)

    fig, ax = plt.subplots(dpi=300, figsize=(8, 4))
    # ax.plot(x, avg_support_active, label="avg. support active cells (without inhibition)")
    # ax.fill_between(x, min_support_active, max_support_active, color='b', alpha=.15,
    #                 label="min/max support (without inhibition)")

    ax.plot(x, avg_support_active, label="avg. support active cells", color='b')
    ax.fill_between(x, min_support_active, max_support_active, color='b', alpha=.15,
                    label="min/max support")

    # ax.plot(x, [LAMBDA] * len(avg_support_active), color='r', linestyle='--', label="λ")
    ax.plot(x, avg_support_inactive, label="avg. support inactive cells", color='orange')
    # plt.title(title)
    plt.legend()
    plt.ylabel("Support Strength")
    plt.xlabel("Epoch")
    plt.yticks(np.arange(0, X_MAX + 1, 2), list(np.arange(0, X_MAX + 1, 2)))
    d = 2 if len(avg_support_active) <= 50 else 4
    plt.xticks(np.arange(0, len(avg_support_active) + 1, d), list(np.arange(0, len(avg_support_active) + 1, d)))
    plt.xlim(1, len(avg_support_active))
    plt.ylim(0, X_MAX)
    plt.tight_layout()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    for title, run_id in RUN_PATHS.items():
        avg_support_active, min_support_active, max_support_active, avg_support_inactive = get_support_from_wandb(run_id)
        print_support_active_cells(title, min_support_active, max_support_active, avg_support_active, avg_support_inactive)
