#!/bin/bash

#SBATCH --job-name=EvaluationLoadNetUniform
#SBATCH --partition=gpu_a100
#SBATCH --output=output_eval_load_net_uniform.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Marc.Serrano.Altena@gmail.com

singularity exec --nv \
    cluster_image/cna.sif \
    python3 src/main_evaluation.py net-fragments_uniform\
        --load "checkpoints/net-fragments_uniform.ckpt" \
        --noise 0 \
        --line_interrupt 0 \
        --load_baseline_activations_path "activations/net-fragments_uniform_baseline_activations.pt" \
        --act_threshold 0.5 \