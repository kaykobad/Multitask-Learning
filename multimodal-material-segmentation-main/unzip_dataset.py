import zipfile


with zipfile.ZipFile("datasets/NYUDv2-SUNRGBD.zip", "r") as zip_ref:
    zip_ref.extractall("datasets/")