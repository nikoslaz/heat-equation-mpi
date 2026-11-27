"""Generates colab/heat_cuda.ipynb, a Google Colab notebook that clones this repository, builds it with CMake,
runs the tests (on the GPU), benchmarks the CUDA programs and draws the charts.

Run it again after changing the notebook's steps:  python3 colab/make_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/nikoslaz/heat-equation-mpi.git"


def lines(text):
    """Notebook cell source: a list of lines, each ending in a newline except the last."""
    parts = text.strip("\n").split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(text)}


cells = [
    markdown("""
# 2D heat equation on a GPU

This notebook builds [heat-equation-mpi](https://github.com/nikoslaz/heat-equation-mpi) on a free Colab GPU,
runs its tests, benchmarks the CUDA kernels and draws the charts. It takes about five minutes.

1. **Runtime → Change runtime type → T4 GPU** (any NVIDIA GPU works).
2. **Runtime → Run all**.

To try your own fork or branch, change `REPO` and `BRANCH` in the first cell.
"""),
    markdown("## Get the code and the tools"),
    code(f"""
REPO = "{REPO}"
BRANCH = "main"

import os, shutil
# Colab runs as root, which Open MPI refuses unless told it is fine.
os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
os.environ["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"

if not os.path.isdir("/content/heat/.git"):
    !git clone --quiet --depth 1 --branch {{BRANCH}} {{REPO}} /content/heat
%cd /content/heat
if shutil.which("mpicxx") is None:
    !apt-get -qq update > /dev/null && apt-get -qq install -y openmpi-bin libopenmpi-dev > /dev/null
!nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
!nvcc --version | tail -1
!cmake --version | head -1
!mpirun --version | head -1
"""),
    markdown("## Build"),
    code("""
cap = !nvidia-smi --query-gpu=compute_cap --format=csv,noheader
arch = cap[0].strip().replace(".", "")  # e.g. 7.5 -> 75 for a T4
!cmake --preset cuda -DCMAKE_CUDA_ARCHITECTURES={arch} > /dev/null
!cmake --build --preset cuda -j 4 2>&1 | grep -E "error|warning|Built target"
"""),
    markdown("""
## Tests

Every program runs with `--verify`, which compares every grid point and the average with a serial CPU run.
The CUDA tests cover each kernel in double and float precision, and `heat_mpi_cuda` with up to six
processes sharing the GPU.
"""),
    code("""
!ctest --preset cuda -j 2 > ctest.log 2>&1; grep -E "tests passed|tests failed|\\(Failed\\)|\\(Timeout\\)" ctest.log
# If anything failed, run just those tests again and show their output:
!grep -q "tests failed" ctest.log && ctest --preset cuda --rerun-failed --output-on-failure 2>&1 | tail -n 60
"""),
    markdown("## Benchmark"),
    code("!scripts/bench_cuda.sh build/cuda"),
    markdown("## Charts"),
    code("""
from IPython.display import Image, display
!python3 scripts/make_charts.py
for chart in ["performance", "gpu-kernels"]:
    path = f"docs/images/{chart}-light.png"
    if os.path.exists(path):
        display(Image(path))
"""),
    markdown("## The temperature after 1000 steps"),
    code("""
!build/cuda/heat_cuda 1000 256 --kernel temporal --csv grid.csv
!python3 scripts/plot_grid.py grid.csv grid.png > /dev/null
display(Image("grid.png"))
"""),
    markdown("""
## Download the measurements

The benchmark wrote `results/gpu-<your GPU>.csv`. Put it in the repository's `results/` directory and run
`python3 scripts/make_charts.py` to update the charts in the README.
"""),
    code("""
import glob
from google.colab import files
for path in glob.glob("results/gpu-*.csv"):
    files.download(path)
"""),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

out = ROOT / "colab" / "heat_cuda.ipynb"
out.write_text(json.dumps(notebook, indent=1) + "\n")
print("Wrote", out.relative_to(ROOT))
