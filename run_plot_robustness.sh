#!/bin/bash

#SBATCH --job-name=PlotRobustness
#SBATCH --partition=gpu_a100
#SBATCH --output=output_plot_robustness.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Marc.Serrano.Altena@gmail.com

singularity exec --nv \
    cluster_image/cna.sif \
    python3 src/plot_robustness.py