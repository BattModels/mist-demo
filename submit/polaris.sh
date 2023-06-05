#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -q debug
#PBS -A Catalyst

# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1

# Change to working directory
cd ${PBS_O_WORKDIR}

module load singularity/3.8.7

# PATH to the Container
CONTAINER="/lus/swift/home/awadell/training_container.sif"

# MPI and OpenMP settings
NNODES=$(wc -l <$PBS_NODEFILE)
NRANKS_PER_NODE=$(nvidia-smi -L | wc -l)
NDEPTH=8
NTHREADS=1

NTOTRANKS=$((NNODES * NRANKS_PER_NODE))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

# For applications that need mpiexec to bind MPI ranks to GPUs
mpiexec \
	-n ${NTOTRANKS} \
	--ppn ${NRANKS_PER_NODE} \
	--depth=${NDEPTH} \
	--cpu-bind depth \
	--env OMP_NUM_THREADS=${NTHREADS} \
	-env OMP_PLACES=threads \
	./set_affinity_gpu_polaris.sh \
	singularity run -B /lus:/lus --nv ${CONTAINER} python train.py --trainer.fast_dev_run 2
