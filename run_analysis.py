"""Run the full GreedGauge analysis using the compressed trade dataset."""

import gzip
import os
import runpy
import shutil

SOURCE = "compressed_data.csv.gz"
TEMP = "compressed_data.csv"

if not os.path.exists(SOURCE):
    raise FileNotFoundError(f"Missing dataset: {SOURCE}")

try:
    with gzip.open(SOURCE, "rb") as src, open(TEMP, "wb") as dst:
        shutil.copyfileobj(src, dst)
    runpy.run_path("crypto.py", run_name="__main__")
finally:
    if os.path.exists(TEMP):
        os.remove(TEMP)
