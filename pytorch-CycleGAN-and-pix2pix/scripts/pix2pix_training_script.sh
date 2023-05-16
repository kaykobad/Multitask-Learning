#!/bin/bash -l

#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --mail-user=mreza025@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH -p batch
#SBATCH --output=output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged
#SBATCH --gres=gpu:1

conda activate pix2pix

python train.py \
--dataroot ./datasets/PBVS-23-Data/ \
--name Sar2Eo-1 \
--model pix2pix \
--netG unet_256 \
--n_epochs 100 \
--direction AtoB \
--lambda_L1 100 \
--dataset_mode aligned \
--norm batch \
--pool_size 0 \
--input_nc 1 \
--output_nc 1 \
--wandb_project_name SAR-2-EO-Translation \
--use_wandb 