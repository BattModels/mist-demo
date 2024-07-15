# Pre-training MIST (Molecular Insight SMILES Transformer)
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

4. Install `ipykernel` and add a kernel for the environment.
```shell
pip install ipykernel
python -m ipykernel install --user --name mist_demo
```

## Using the notebooks

The notebooks demonstrating the MIST pre-training workflow are in the `notebooks` directory. To run them:
1. Activate the environment
```shell
# On Polaris
# Initialize environment
module purge
module use /soft/modulefiles/
module --ignore_cache load conda/2024-04-29 gcc-native/12.3 PrgEnv-nvhpc
export CC=gcc-12
export CXX=g++-12
conda activate base
source ./mist/activate
```
2. Request an interactive session with one GPU node.
```
qsub -I -l select=1 -l filesystems=[home:filesystem] -l walltime=01:00:00 -q debug -A [AccountName]
```
4. Launch a `jupyter notebook`  server and select the `mist_env` kernel.
```
jupyter notebook --ip $(hostname) --no-browser
```

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

## Checkpoint

A sample checkpoint is also available on Dropbox. This data should be downloaded and placed in the `sample_checkpoint` folder.
