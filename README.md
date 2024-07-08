# Pre-training MIST (Molecular Insight Transformer)
This repository is an example of the pre-training workflow for a transformers trained on molecular datasets.

# Installation

## Polaris

1. Install [rust](https://www.rust-lang.org/tools/install)

2. Load conda
```shell
module purge
module use /soft/modulefiles/
module --ignore_cache load conda/2024-04-29
conda activate base
```

3. Install poetry + pipx
```shell
python -m pip install pipx
python -m pipx ensurepath
python -m pipx install maturin
python -m pipx install --python $(which python) poetry
```

4. Install environment: `poetry install`

# Data
