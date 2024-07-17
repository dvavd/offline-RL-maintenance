#!/bin/bash
#SBATCH --job-name=dataset_sampling
#SBATCH --output=result_%A_%a.out
#SBATCH --error=error_%A_%a.err
#SBATCH --array=0-24
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8192

# Initialise Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

# Compute optimality and seed based on SLURM_ARRAY_TASK_ID
let "optimality_index = $SLURM_ARRAY_TASK_ID / 5"
let "seed = $SLURM_ARRAY_TASK_ID % 5 + 1"
optimality_values=(0.0 0.25 0.50 0.75 1.0)
optimality=${optimality_values[$optimality_index]}

# Run Python script with computed optimality and seed
python data_sampling.py --randomness $optimality --seed $seed --num_trajectories 100000
