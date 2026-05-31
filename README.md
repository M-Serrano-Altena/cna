# The Cooperative Network Architecture — Extended

This repository is an extension of the original [CNA repository](https://github.com/sagerpascal/cna/tree/main) by Pascal Sager, accompanying the paper [The Cooperative Network Architecture: Learning Structured Networks as Representation of Sensory Patterns](https://arxiv.org/pdf/2407.05650). It contains the original code for reproducing the paper's results, as well as additional experiments aimed at making the S1 feature extraction layer trainable.

For the original repository and documentation, see [https://github.com/sagerpascal/cna](https://github.com/sagerpascal/cna).

## Setup
Create conda environment

```bash
conda create --name net-fragments python=3.10
```

Activate environment

```bash
conda activate net-fragments
```

Install requirements

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Run experiments

### Reproducing the original results

Training can be done as follows:

```bash
python main_training.py net-fragments --wandb --plot --store 
```

where `net-fragments` is the name of the training configuration, `store_path` is the path where the results should be stored, and `wandb` and `plot` are optional flags to enable logging to [wandb](https://wandb.ai/) and plotting of the results, respectively.

#### Evaluation

For evaluation, first create the baseline activations of the trained model:

```bash
python main_evaluation.py net-fragments --load  --noise 0 --line_interrupt 0 --store_baseline_activations_path 
```

Then run the evaluation:

```bash
python main_evaluation.py net-framents --load  --noise  --line_interrupt  --load_baseline_activations_path  --act_threshold  --square_factor  --wandb
```

where `noise` is the noise level, `line_interrupt` is the number of interrupted lines, `act_threshold` is the activation threshold, and `square_factor` is the square factor.
You can also use `--act_threshold bernoulli` to test with Bernoulli neurons. However, in this case, the results are based on randomness and will vary between runs. Therefore, we recommend using a fixed activation threshold, e.g. `--act_threshold 0.5`, which also makes the plots easier to comprehend.

You can replace the `net-fragments` config with `autoencoder` to train and evaluate using the autoencoder model instead

### Extension: learnable S1 layer

To make the S1 feature extraction layer trainable, set `learn_s1: True` in the config file (or use `net-fragments_learn_s1` as the config). When enabled, the S1 kernels are first trained to match the activations of the original fixed kernels, after which the learned weights are fixed and S2 is trained as normal. This is currently implemented as a proof of concept, though the approach did not fully succeed.


### Extension: uniform training data

To test whether a skewed distribution of line orientations could explain the observed asymmetry in S2 channel activations, the data generation was changed from random to purely uniform sampling across the 300 training images. This can be reproduced using the `net-fragments_uniform` config, which sets `uniform_sampling: True`. The asymmetry was found to persist regardless, though the model performed slightly worse overall.