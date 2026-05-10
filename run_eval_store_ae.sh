#!/bin/bash

#SBATCH --job-name=EvaluationStoreAE
#SBATCH --partition=gpu_a100
#SBATCH --output=output_eval_store_ae.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Marc.Serrano.Altena@gmail.com

singularity exec --nv \
    cluster_image/cna.sif \
    python3 src/main_evaluation.py autoencoder\
        --load "checkpoints/autoencoder.ckpt" \
        --noise 0 \
        --line_interrupt 0 \
        --store_baseline_activations_path "activations/autoencoder_baseline_activations.pt" \