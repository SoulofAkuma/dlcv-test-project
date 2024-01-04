#!/bin/sh
#SBATCH --job-name=DLCV-Test-Project-7715464
#SBATCH --output=/scratch/vihps/vihps01/stdouts/%j.out
#SBATCH --error=/scratch/vihps/vihps01/stderrs/%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --mail-user=vihps01
#SBATCH --mail-type=NONE

conda init
conda activate /scratch/vihps/vihps01/env/

srun python3 /home/vihps/vihps01/dlcv-test-project/run.py