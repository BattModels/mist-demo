#!/bin/bash
#SBATCH -N {{ nodes or 1 }}
#SBATCH -p RM
#SBATCH --time 4:00:00
#SBATCH -A {{ account or "che210007p" }}
# Load spark and cluster

ml load spark
source activate

# Set spark python
export PYSPARK_PYTHON="$(realpath $(pwd))/.venv/bin/python"
export PYSPARK_DRIVER_PYTHON="${PYSPARK_PYTHON}"

# Submit script
spark-submit \
   "$(realpath $(pwd))/electrolyte_fm/tokenize/process.py" \
    split \
    "$(realpath ~/che210007p/shared/REALSpace_t2/)" \
    "$(realpath ~/che210007p/shared)/realspace_v2/"
