#! /bin/bash
#SBATCH -J Facke_ILVR
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -o log.out
#SBATCH -t 10-10:00:00
#SBATCH --gres=gpu:1
module load anaconda3/2019.07
source activate pytorch_1.11
python -u demo_ILVR.py --attention_resolutions 16 --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown --use_scale_shift_norm --timestep_respacing 100 --DDPM_pth ./checkpoints/Facke_finetune_DDPM/DDPM/latest.pth --down_N 32 --range_t 20 --clip_denoised --batchSize 8 --name Facke_finetune_DDPM_lr5e-5
