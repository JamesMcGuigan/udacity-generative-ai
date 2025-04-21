# https://filesystem-spec.readthedocs.io/en/latest/usage.html

import fsspec
fs    = fsspec.filesystem('file')
# files = fs.ls("./input/", detail=True)
files = fs.glob("./input/*.png")
for file in files: print(file)