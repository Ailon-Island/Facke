#! /bin/bash
#SBATCH -J Facke
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -o log.out
#SBATCH -t 10-10:00:00
#SBATCH --gres=gpu:2
module load anaconda3/2019.07
source activate pytorch_1.11
python -u ./train_SimSwap.py --batchSize 32 --num_workers 64 --debug
