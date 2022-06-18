#! /bin/bash
#SBATCH -J Facke_nFM
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -o log.out
#SBATCH -t 10-10:00:00
#SBATCH --gres=gpu:1
module load anaconda3/2019.07
source activate pytorch_1.11
python -u ./benchmark.py --batchSize 32 --nThreads 32 --name SimSwap_WO_intra-ID_random_nFM --no_intra_ID_random
