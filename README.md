# Electrolyte Foundation Model
Benchmarking RoBERTa model pre-training on molecular datasets.

# Installation

## Polaris

1. Install [rust](https://www.rust-lang.org/tools/install)

2. Load conda
```shell
module purge
module use /soft/modulefiles/
module --ignore_cache load conda/2024-04-29
module conda/2024-04-25
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

# Submitting Jobs

```shell
source activate # Activate Environment
./submit/submit.py ./submit/polaris.j2 | qsub
```

See `submit/submit.py --help` for more info

# Development

## Pre-commit

We use [pre-commit](https://pre-commit.com) to preform various linting checks on the code. To enable:

1. Install poetry (See above)
2. Run pre-commit: `pre-commit`
3. Run before committing: `pre-commit install --allow-missing-config`
