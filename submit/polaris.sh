#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -q debug
#PBS -A fm_electrolyte

# Change to working directory (Git Working Directory)
#cd ${PBS_O_WORKDIR}
echo "PWD: $(pwd)"

# Load Modules
module load e4s/22.05/PrgEnv-gnu
module load openmpi/4.1.3
module load singularity/3.8.7

# Enable internet
source ./submit/proxy_settings.sh

export MASTER_ADDR=localhost
export MASTER_PORT=2345

# MPI and OpenMP settings
NNODES=$(wc -l <$PBS_NODEFILE)
NNODES=1
NRANKS_PER_NODE=4
NDEPTH=64

NTOTRANKS=$((NNODES * NRANKS_PER_NODE))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}"
echo <$PBS_NODEFILE

# Logging
echo "$(df -h /local/scratch)"

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=PHB

# Launch Training
pwd
mpiexec \
	-n ${NTOTRANKS} \
	bash ./submit/run-polaris-generation.sh \
	singularity run \
	--nv \
	./training_container.sif python train.py --trainer.fast_dev_run 1
