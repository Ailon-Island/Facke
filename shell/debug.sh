#! /bin/bash
#SBATCH -J Facke_dbg
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -o log.out
#SBATCH -t 10-10:00:00
#SBATCH --gres=gpu:2
module load anaconda3/2019.07
conda init
conda activate pytorch_1.11
python -u ./train_SimSwap.py --batchSize 4 --debug --nThreads 32 --ID_check
