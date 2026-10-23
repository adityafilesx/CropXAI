#!/bin/bash
#SBATCH --job-name=cropxai-campaign
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm/cropxai_%A_%a.out
#SBATCH --error=logs/slurm/cropxai_%A_%a.err

# Load modules (Adjust for Jean Zay / University Cluster)
# module load python/3.10
# module load cuda/11.8

# Activate virtual environment
# source venv/bin/activate

echo "Starting SLURM job array task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# The SLURM executor will dynamically append the python command here:
