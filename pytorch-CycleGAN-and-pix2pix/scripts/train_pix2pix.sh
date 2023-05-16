set -ex
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0




python train.py --dataroot ./datasets/translation-data/train --name Sar2Eo --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --input_nc 1 --output_nc 1