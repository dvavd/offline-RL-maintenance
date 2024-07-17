#!/bin/bash
#SBATCH --job-name=evaluation_fqe
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=0-14
#SBATCH --time=250:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8192

# Initialise Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate training

# Define the arrays for optimality and seeds
optimality_values=(0 25 50 75 100)
seeds=(1 2 3)

# Compute indexes based on SLURM_ARRAY_TASK_ID
let "optimality_index = $SLURM_ARRAY_TASK_ID / 3"
let "seed_index = $SLURM_ARRAY_TASK_ID % 3"

# Assign the values based on computed indexes
optimality=${optimality_values[$optimality_index]}
seed=${seeds[$seed_index]}

# Run Python script with computed optimality and seed
python evaluation_fqe.py --optimalities $optimality --seed $seed
