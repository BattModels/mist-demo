#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -q debug
#PBS -A fm_electrolyte

# Change to working directory (Git Working Directory)
cd ${PBS_O_WORKDIR}
echo "PWD: $(pwd)"

# Load Modules
module load e4s/22.05/PrgEnv-gnu
module load openmpi/4.1.3
module load singularity/3.8.7

# Enable internet
source ./submit/proxy_settings.sh

# MPI and OpenMP settings
NNODES=$(wc -l <$PBS_NODEFILE)
NRANKS_PER_NODE=1
NDEPTH=64

NTOTRANKS=$((NNODES * NRANKS_PER_NODE))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}"
echo <$PBS_NODEFILE

# Logging
echo "$(df -h /local/scratch)"

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=PHB

env

# Launch Training
mpiexec \
	-n ${NTOTRANKS} \
	--ppn ${NRANKS_PER_NODE} \
	--depth=${NDEPTH} \
	--cpu-bind none \
	--mem-bind none \
	--hostfile $PBS_NODEFILE \
	./submit/set_affinity_gpu_polaris.sh \
	singularity run -B /lus:/lus --nv ./training_container.sif python train.py fit --trainer.max_epochs 2
