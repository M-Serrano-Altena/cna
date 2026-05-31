#!/bin/bash

#SBATCH --job-name=EvaluationStoreNetUniform
#SBATCH --partition=gpu_a100
#SBATCH --output=output_eval_store_net_uniform.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Marc.Serrano.Altena@gmail.com

singularity exec --nv \
    cluster_image/cna.sif \
    python3 src/main_evaluation.py net-fragments_uniform\
        --load "checkpoints/net-fragments_uniform.ckpt" \
        --noise 0 \
        --line_interrupt 0 \
        --act_threshold 0.7 \
        --square_factor 1.8 1.9 2.0 2.1 2.2 2.3 \
        --store_baseline_activations_path "activations/net-fragments_uniform_baseline_activations.pt" \