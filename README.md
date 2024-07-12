# Pre-training MIST (Molecular Insight Transformer)
This repository is an example of the pre-training workflow for a transformer trained on molecular datasets.

# Installation

MIST is trained primarily on Polaris, installation instructions for this system are provided here. 
Installation may be slightly different for other systems.

## Polaris

1. Load conda
```shell
module purge
module use /soft/modulefiles/
module --ignore_cache load conda/2024-04-29
conda activate base
```

2. Install poetry + pipx
```shell
python -m pip install pipx
python -m pipx ensurepath
python -m pipx install --python $(which python) poetry
```

3. Install environment: `poetry install`

## Data

The pre-training data is available on [Dropbox](https://www.dropbox.com/scl/fo/3z1lklbper07ojtp5t4iu/AHUEJ_3j5_CRVpWmcGLW3kQ?rlkey=2818imymvf5mk5byz0c7ei1ij&dl=0).
This data should be downloaded and extracted in the `sample_data` folder. It requires ~2.2GB of disk space.

```
sample_data
├── data
│   ├── train
│   │   ├── xaaa.txt
│   │   ├── xaab.txt
│   │   ├── ...
│   ├── test
│   │   ├── xaaa.txt
│   │   ├── xaab.txt
│   │   ├── ...
│   ├── val
│   │   ├── xaaa.txt
│   │   ├── xaab.txt
│   │   ├── ...
```

The data is pre-shuffled and split into training, validation and test sets with a 80:20:20 ratio. 
The training dataset has `~0.25B` molecules, while the test and validation sets have `62M` molecules each.