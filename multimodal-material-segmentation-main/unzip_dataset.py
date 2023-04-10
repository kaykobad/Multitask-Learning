import zipfile


with zipfile.ZipFile("datasets/multimodal_dataset.zip", "r") as zip_ref:
    zip_ref.extractall("datasets/multimodal_dataset")