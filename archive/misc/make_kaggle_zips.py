# make_kaggle_zips.py
from pathlib import Path
import zipfile

def zip_folder(folder, output):
    folder = Path(folder)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for f in folder.rglob("*"):
            if f.is_file():
                z.write(
                    f,
                    arcname=f.relative_to(folder.parent).as_posix()
                )

zip_folder("datasets/pope", "pope_datasets.zip")
zip_folder("src", "ugaa_src.zip")
zip_folder("analysis", "ugaa_analysis.zip")

print("Done")