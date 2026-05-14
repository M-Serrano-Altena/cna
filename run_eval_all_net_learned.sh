#!/bin/bash

#SBATCH --job-name=EvaluationAllNetLearned
#SBATCH --partition=gpu_a100
#SBATCH --output=output_eval_all_net_learned.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=12G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Marc.Serrano.Altena@gmail.com

echo "Bash version ${BASH_VERSION} - Evaluate all models..."

SIF_IMAGE=${SIF_IMAGE:-cluster_image/cna.sif}
SINGULARITY_CMD=${SINGULARITY_CMD:-singularity exec --nv "$SIF_IMAGE"}
ACTIVATIONS_DIR=${ACTIVATIONS_DIR:-activations}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-checkpoints}

for config in 'net-fragments_learn_s1'; do

  for act_threshold in 0.3 0.5 0.7; do

    for square_factor in '0.6 0.8 1.0 1.2 1.4 1.6' '1.2 1.4 1.6 1.8 2.0 2.2' '1.8 1.9 2.0 2.1 2.2 2.3'; do

      echo "Store baseline $config..."
      $SINGULARITY_CMD python src/main_evaluation.py $config --load $CHECKPOINTS_DIR/$config.ckpt --noise 0 --line_interrupt 0 --store_baseline_activations_path $ACTIVATIONS_DIR/${config}_baseline_activations.pt --act_threshold $act_threshold --square_factor $square_factor
      sleep 2

      echo "Evaluate different noise for $config..."
      for noise in $(seq 0.0 .01 0.2); do
        $SINGULARITY_CMD python src/main_evaluation.py $config --load $CHECKPOINTS_DIR/$config.ckpt --noise $noise --line_interrupt 0 --load_baseline_activations_path $ACTIVATIONS_DIR/${config}_baseline_activations.pt --act_threshold $act_threshold --square_factor $square_factor
      done

      echo "Evaluate different line interrupts for $config..."
      for li in {1..7}; do
        $SINGULARITY_CMD python src/main_evaluation.py $config --load $CHECKPOINTS_DIR/$config.ckpt --noise 0 --line_interrupt $li --load_baseline_activations_path $ACTIVATIONS_DIR/${config}_baseline_activations.pt --act_threshold $act_threshold --square_factor $square_factor
      done

    done
  done
done