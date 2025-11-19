# MIST Demo

This repository contains tutorials for fine-tuning and applying MIST (Molecular Insight SMILES Transformer) foundation models to chemical problems. 
Model checkpoints for MIST models are available on [HuggingFace](https://huggingface.co/mist-models) and on Zenodo.
The full code, including pre-training, model development and full scale application demos can be found in the [`mist`](https://github.com/BattModels/mist-demo) repository.

# Tutorials

#### [run_finetuning.ipynb](tutorials/run_finetuning.ipynb)
Complete fine-tuning workflow for MIST encoder models:
- Finetuning with **LoRA** (Low-Rank Adaptation) for parameter-efficient training
- Hyperparameter optimization
- Training on the QM9 dataset for molecular property prediction
- Model evaluation 

#### [tutorials/molecular_property_prediction.ipynb](tutorials/molecular_property_prediction.ipynb)
Inference demonstrations using fine-tuned MIST models:
- Loading pretrained MIST checkpoints from HuggingFace
- Predicting boiling point, flash point, and melting point
- Analyzing property trends for alkenes and alcohols

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mist-demo
```

2. Create a virtual environment and install dependencies using [uv](https://docs.astral.sh/uv/getting-started/installation/)
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Running the Notebooks

Launch Jupyter and open any notebook in `mist-demo/tutorials`:
```bash
jupyter notebook
```