#!/bin/bash

#SBATCH --job-name=TrainingNetLearned
#SBATCH --partition=gpu_a100
#SBATCH --output=output_training_net_learned.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=12G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Marc.Serrano.Altena@gmail.com

singularity exec --nv \
    cluster_image/cna.sif \
    python3 src/main_training.py net-fragments_learn_s1\
        --plot \
        --store "checkpoints/net-fragments_learn_s1.ckpt" \