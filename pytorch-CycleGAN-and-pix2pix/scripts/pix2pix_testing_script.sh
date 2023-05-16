set -ex
python test.py \
--dataroot ./datasets/translation-data/test/ \
--name Sar2Eo-1 \
--epoch 100 \
--model test \
--netG unet_256 \
--direction BtoA \
--dataset_mode single \
--norm batch \
--input_nc 1 \
--output_nc 1 

