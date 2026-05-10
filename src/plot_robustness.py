import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Union, Optional

AUTOENCODER_RESULTS_DIR = "results/autoencoder"
GENERAL_RESULTS_DIR = "results/"


def replace_square_list(sl: list) -> str:
    """Convert a linear list to a mathematical expression string.
    
    Validates that the input list represents a linear sequence and converts it
    to a string representation in the form "base + delta*t". The function calculates
    the base value and delta (step size) and ensures all elements follow the linear
    pattern before returning the formatted string.
    
    Args:
        sl (list): A list of numeric values representing a linear sequence.
        
    Returns:
        str: A formatted string representation of the linear sequence as "base + delta*t".
        
    Raises:
        AssertionError: If the list does not represent a valid linear sequence.
    """
    base_value = sl[0]
    delta = round(sl[1] - sl[0], 3)

    # check if list is linear
    for i in range(1, len(sl)):
        assert sl[i] == round(base_value + i * delta, 3), "List is not linear"

    return "{} + {}t".format(base_value, delta)


def get_data(file_path: Path) -> pd.DataFrame:
    """Load and parse experiment results from a JSON file into a DataFrame.
    
    Reads a JSON lines file where each line contains experiment metadata and metrics.
    Extracts configuration parameters (noise, line_interrupt), noise reduction metrics,
    reconstruction accuracy/recall/precision, and optional lateral model parameters.
    Returns a pandas DataFrame with all extracted features.
    
    Args:
        file_path (Path): Path object pointing to the JSON lines file containing results.
        
    Returns:
        pd.DataFrame: DataFrame containing parsed experiment results with columns for
            noise, line_interrupt, noise_reduction, reconstruction metrics, and optional
            activation bias and square factor parameters.
    """
    results = []
    with open(str(file_path.absolute())) as f:
        filecontents = f.readlines()
        for entry in filecontents:
            data = json.loads(entry)
            result = {
                'noise': data['config']['noise'],
                'line_interrupt': data['config']['line_interrupt'],
                'noise_reduction': data['noise_reduction'],
                'avg_line_recon_accuracy_meter': data['avg_line_recon_accuracy_meter'],
                'avg_line_recon_accuracy_meter_2': (data['avg_line_recon_accuracy_meter'] - 0.75) / 0.25,
                'recon_accuracy': data['recon_accuracy'],
                'recon_recall': data['recon_recall'],
                'recon_precision': data['recon_precision'],
            }

            if 'lateral_model' in data['config']:
                result['act_bias'] = round(float(data['config']['lateral_model']['s2_params']['act_threshold']), 2)
                result['square_factor'] = replace_square_list(data['config']['lateral_model']['s2_params']['square_factor'])

            results.append(result)
    return pd.DataFrame(results)


def feature_noise_to_location_noise(feature_noise: float, round_: bool = False) -> Union[float, np.ndarray]:
    """Convert feature-level noise probability to spatial location noise probability.
    
    Calculates the probability that noise occurs at any spatial location given that
    noise can independently affect each of 4 feature channels. Uses the complement
    probability formula: P(at least one noise) = 1 - (1 - p)^4.
    
    Args:
        feature_noise (float): Probability of noise in a single feature channel (0-1).
        round_ (bool, optional): Whether to round the result to 2 decimal places. Defaults to False.
        
    Returns:
        float | np.ndarray: Spatial location noise probability. Returns rounded values if
            round_=True, otherwise returns float or array of floats matching input shape.
    """
    # calculate probability of noise at each spatial location (can occur at each of the 4 feature channels)
    result = 1 - (1 - feature_noise) ** 4
    if round_:
        result = np.round(result, 2)
    return result


def plot_line(data: pd.DataFrame, x_key: str, x_label: str, y_key: str, y_label: str, z_key: str, z_label: str,
              plot_key: str, plot_label: str, xmin: float, xmax: float, ymin: float, ymax: float,
              x2_func=None, x2_label: str | None = None, set_title: bool = True, filename: str | None = None,
              data_ae: pd.DataFrame | None = None) -> None:
    """Create multi-panel line plots comparing model performance across conditions.
    
    Generates a figure with multiple subplots, one for each unique value in plot_key.
    Each subplot shows multiple lines corresponding to different z_key values. Optionally
    includes a secondary x-axis with transformed tick labels and overlays autoencoder
    baseline results for comparison.
    
    Args:
        data (pd.DataFrame): Main dataset containing experimental results.
        x_key (str): Column name for x-axis values.
        x_label (str): Label for primary x-axis.
        y_key (str): Column name for y-axis values.
        y_label (str): Label for y-axis.
        z_key (str): Column name for line grouping (creates separate lines).
        z_label (str): Label prefix for legend entries from z_key.
        plot_key (str): Column name for subplot creation (one subplot per unique value).
        plot_label (str): Label prefix for subplot titles.
        xmin (float): Minimum x-axis limit.
        xmax (float): Maximum x-axis limit.
        ymin (float): Minimum y-axis limit.
        ymax (float): Maximum y-axis limit.
        x2_func (callable, optional): Function to transform x-axis values for secondary axis. Defaults to None.
        x2_label (str | None, optional): Label for secondary x-axis. Defaults to None.
        set_title (bool, optional): Whether to set subplot titles. Defaults to True.
        filename (str | None, optional): Path to save figure. Defaults to None.
        data_ae (pd.DataFrame | None, optional): Autoencoder baseline data to overlay. Defaults to None.
    """

    # if problems with font, run it on local machine
    plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 14

    fig, axs = plt.subplots(ncols=len(data[plot_key].unique()), figsize=(13, 4), dpi=300)

    

    for ic, (ax, pk) in enumerate(zip(axs, sorted(data[plot_key].unique()))):
        data_ = data[data[plot_key] == pk]

        z_values = list(data_[z_key].unique())
        z_values = sorted(z_values)

        for zv in z_values:
            z = data_[data_[z_key] == zv]
            ax.plot(z[x_key].values, z[y_key].values, label="{}{}".format(z_label, zv))

        if data_ae is not None:
            ax.plot(data_ae[x_key].values, data_ae[y_key].values, label="AE Baseline", linestyle='--', c='red')

        if set_title:
            ax.set_title("{}{}".format(plot_label, pk))

        ax.set_xlabel(x_label)
        if 'noise' in x_key.lower():
            ax.set_xticklabels(feature_noise_to_location_noise(ax.get_xticks(), round_=True))
        if ic == 0:
            ax.set_ylabel(y_label)

        ax.set_ylim(ymax=ymax, ymin=ymin)
        ax.set_xlim(xmin=xmin, xmax=xmax)

        if x2_func is not None:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(ax.get_xticks()[1:-1])
            ax2.set_xticklabels(x2_func(ax.get_xticks()[1:-1], round_=True))
            ax2.set_xlabel(x2_label)

        ax.legend()
        ax.grid()

    plt.tight_layout()
    if filename is not None:
        plt.savefig(f"{GENERAL_RESULTS_DIR}/{filename}")
    plt.show()


def plot(data: pd.DataFrame, configname: str, data_ae: Optional[pd.DataFrame] = None) -> None:
    """Generate comprehensive robustness plots for noise and line interrupt conditions.
    
    Creates six subplots organized by configuration: three for noise robustness
    (noise_reduction, recall, precision) and three for line interrupt robustness
    (feature reconstruction rate, recall, precision). Each set compares different
    activation bias and power factor parameters against an autoencoder baseline.
    
    Args:
        data (pd.DataFrame): Experimental results DataFrame from get_data().
        configname (str): Configuration name used for organizing output directories.
        data_ae (pd.DataFrame, optional): Autoencoder baseline results for comparison.
            Defaults to None.
    """

    def fname(x):
        return f"{configname}/{x}.png"


    # cleanup data
    data.loc[data.noise == 0, 'noise_reduction'] = 1.0
    if data_ae is not None:
        data_ae.loc[data_ae.noise == 0, 'noise_reduction'] = 1.0

    # plot noise only
    if data_ae is not None:
        data_ae_1 = data_ae[data_ae['line_interrupt'] == 0]
    else:
        data_ae_1 = None
    data_1 = data[data['line_interrupt'] == 0]
    plot_line(data_1, x_key="noise", x_label="Noise", y_key="noise_reduction", y_label="Noise Reduction Rate",
              z_key='act_bias', z_label='b = ', plot_key='square_factor', plot_label='Power Factor γ = ',
              x2_func=None, x2_label="Spatial Noise", filename=fname("1_noise_reduction"),
              xmin=-0.005, xmax=0.205, ymin=0.795, ymax=1.005, data_ae=data_ae_1)

    plot_line(data_1, x_key="noise", x_label="Noise", y_key="recon_recall", y_label="Recall",
              z_key='act_bias', z_label='b = ', plot_key='square_factor', plot_label='Power Factor γ = ',
              x2_func=None, x2_label="Spatial Noise", set_title=False,
              filename=fname("2_recon_recall"), xmin=-0.005, xmax=0.205, ymin=0.08, ymax=1.02, data_ae=data_ae_1)

    plot_line(data_1, x_key="noise", x_label="Noise", y_key="recon_precision", y_label="Precision",
              z_key='act_bias', z_label='b = ', plot_key='square_factor', plot_label='Power Factor γ = ',
              x2_func=None, x2_label="Spatial Noise", set_title=False,
              filename=fname("3_recon_precision"), xmin=-0.005, xmax=0.205, ymin=0.08, ymax=1.02, data_ae=data_ae_1)

    # plot line interrupt only
    if data_ae is not None:
        data_ae_1 = data_ae[data_ae['noise'] == 0.0]
    else:
        data_ae_1 = None
    data_1 = data[data['noise'] == 0.0]

    # set accuracy to 1 where line is not interrupted for the plot
    data_1.loc[data_1.line_interrupt == 0, 'avg_line_recon_accuracy_meter'] = 1.0
    if data_ae_1 is not None:
        data_ae_1.loc[data_ae_1.line_interrupt == 0, 'avg_line_recon_accuracy_meter'] = 1.0

    plot_line(data_1, x_key="line_interrupt", x_label="Line Interrupt", y_key="avg_line_recon_accuracy_meter",
              y_label="Feature Reconstruction Rate", z_key='act_bias', z_label='b = ',
              plot_key='square_factor', plot_label='Power Factor γ = ', filename=fname("4_avg_line_recon_accuracy"),
              xmin=-0.1, xmax=7.1, ymin=-0.02, ymax=1.02, data_ae=data_ae_1)
    plot_line(data_1, x_key="line_interrupt", x_label="Line Interrupt", y_key="recon_recall", y_label="Recall",
              z_key='act_bias', z_label='b = ', plot_key='square_factor', plot_label='Power Factor γ = ',
              set_title=False, filename=fname("5_recon_recall"), xmin=-0.1, xmax=7.1, ymin=0.49, ymax=1.01, data_ae=data_ae_1)
    plot_line(data_1, x_key="line_interrupt", x_label="Line Interrupt", y_key="recon_precision", y_label="Precision",
              z_key='act_bias', z_label='b = ', plot_key='square_factor', plot_label='Power Factor γ = ',
              set_title=False, filename=fname("6_recon_precision"), xmin=-0.1, xmax=7.1, ymin=0.49, ymax=1.01, data_ae=data_ae_1)


if __name__ == '__main__':
    data_ae = get_data(Path(AUTOENCODER_RESULTS_DIR) / "experiment_results.json")
    for f in ['net-fragments']:
        file_path = Path(GENERAL_RESULTS_DIR) / f / "experiment_results.json"
        data = get_data(file_path)
        plot(data, f, data_ae)
