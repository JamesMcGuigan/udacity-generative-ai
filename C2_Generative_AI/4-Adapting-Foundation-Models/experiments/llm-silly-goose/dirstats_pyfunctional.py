# https://github.com/EntilZha/PyFunctional
# https://chatgpt.com/c/67fa909e-3124-8004-9083-55e0728b4980
import json
import os
import fsspec
from PIL import Image
from functional import seq

def main():
    fs    = fsspec.filesystem('file')
    files = fs.glob("./input/*.png")
    stats = (
        seq(files)
        .map(get_image_stats)
        .to_list()
    )
    print(stats)
    with open('./output/fsstats_pyfunctional.txt', 'w') as json_file:
        json.dump(stats, json_file, indent=4)

def get_image_stats(filename):
    with Image.open(filename) as img:
        return {
            "file":   filename,
            "width":  img.width,
            "height": img.height,
            "size":   os.path.getsize(filename)
        }

if __name__ == "__main__":
    main()

