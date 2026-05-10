#!/bin/bash

#SBATCH --job-name=TrainingNet
#SBATCH --partition=gpu_a100
#SBATCH --output=output_training_net.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Marc.Serrano.Altena@gmail.com

singularity exec --nv \
    cluster_image/cna.sif \
    python3 src/main_training.py net-fragments\
        --plot \
        --store "checkpoints/net-fragments.ckpt" \