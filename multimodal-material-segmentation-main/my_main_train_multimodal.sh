#!/bin/bash -l

#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --mail-user=mreza025@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH -p batch
#SBATCH --output=output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged
#SBATCH --gres=gpu:2

conda activate mml

CUDA_VISIBLE_DEVICES=0,1 python my_training_script.py \
  --backbone resnet \
  --lr 0.05 \
  --workers 1 \
  --epochs 1000 \
  --batch-size 8 \
  --ratio 3 \
  --gpu-ids 0,1 \
  --checkname DeepFuseNet \
  --model-name DeepFuseNet-Batch-8-RGB+NIR+Segment \
  --eval-interval 1 \
  --loss-type ce \
  --dataset multimodal_dataset \
  --list-folder list_folder \
  --use-pretrained-resnet \
  --is-multimodal \
  --use-nir \
  --use-segmap 
