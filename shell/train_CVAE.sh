#! /bin/bash
#SBATCH -J Facke_CVAE_New_Generator
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -o log.out
#SBATCH -t 10-10:00:00
#SBATCH --gres=gpu:2
module load anaconda3/2019.07
source activate pytorch_1.11
python -u ./train_CVAE.py --model CVAE --batchSize 32 --name CVAE_GAN --display_freq 2000 --no_intra_ID_random --print_freq 2400
