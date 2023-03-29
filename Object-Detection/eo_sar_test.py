from PIL import Image
from numpy import asarray, min, max, mean

EO_paths = [
    "dataset/train-validation_processed/train/EO/9/EO_30108.png",
    "dataset/train-validation_processed/train/EO/5/EO_75797.png",
    "dataset/train-validation_processed/train/EO/7/EO_15432.png",
    "dataset/train-validation_processed/train/EO/4/EO_11924.png",
    "dataset/train-validation_processed/train/EO/8/EO_24028.png",
]
SAR_paths = [
    "dataset/train-validation_processed/train/SAR/9/SAR_30108.png",
    "dataset/train-validation_processed/train/SAR/5/SAR_75797.png",
    "dataset/train-validation_processed/train/SAR/7/SAR_15432.png",
    "dataset/train-validation_processed/train/SAR/4/SAR_11924.png",
    "dataset/train-validation_processed/train/SAR/8/SAR_24028.png",
]

for i in range(len(EO_paths)):
    eo = Image.open(EO_paths[i])
    sar = Image.open(SAR_paths[i])

    eo = asarray(eo)
    sar = asarray(sar)

    print(f"EO  >==> Shape: {eo.shape}, Min: {min(eo)}, Max: {max(eo)}, Mean: {mean(eo)}, Path:{EO_paths[i]}")
    print(f"SAR >==> Shape: {sar.shape}, Min: {min(sar)}, Max: {max(sar)}, Mean: {mean(sar)}, Path:{SAR_paths[i]}")
    print()