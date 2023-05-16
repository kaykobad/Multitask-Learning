import zipfile


with zipfile.ZipFile("datasets/PBVS-23-Translations/design_data.zip", "r") as zip_ref:
    zip_ref.extractall("datasets/PBVS-23-Translations")