import zipfile

with zipfile.ZipFile("pope_datasets.zip") as z:
    print(z.namelist()[:20])