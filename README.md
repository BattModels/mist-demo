# Electrolyte Foundation Model
Benchmarking RoBERTa model pre-training on molecular datasets.

# Installation

0. Get python3.10

## Polaris
```shell
module conda/2024-04-25
conda activate base
```

1. Install poetry + pipx
```shell
python -m pip install pipx
python -m pipx ensurepath
python -m pipx install --python $(which python) poetry
```

2. Install environment: `poetry install`

# Submitting Jobs

```shell
source activate # Activate Environment
./submit/submit.py ./submit/polaris.j2 | qsub
```

See `submit/submit.py --help` for more info
