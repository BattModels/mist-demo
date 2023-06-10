#!/bin/sh
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=swift:home
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -A fm_electrolyte
#PBS -k doe
#PBS -o /home/awadell/electrolyte_fm/testing-output.out
#PBS -e /home/awadell/electrolyte_fm/testing-error.out

cd $PBS_O_WORKDIR

# Controlling the output of your application
# UG Sec 3.3 page UG-40 Managing Output and Error Files
# By default, PBS spools your output on the compute node and then uses scp to move it the
# destination directory after the job finishes.  Since we have globally mounted file systems
# it is highly recommended that you use the -k option to write directly to the destination
# the doe stands for direct, output, error

set -x

# Internet access on nodes
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3130
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
git config --global http.proxy http://proxy.alcf.anl.gov:3128
echo "Set HTTP_PROXY and to $HTTP_PROXY"

# Set ADDR and PORT for communication
master_node=$(cat $PBS_NODEFILE | head -1)
export MASTER_ADDR=$(host $master_node | head -1 | awk '{print $4}')
export MASTER_PORT=2345

# Enable GPU-MPI (if supported by application)
#export MPICH_GPU_SUPPORT_ENABLED=1

# MPI and OpenMP settings
NNODES=$(wc -l <$PBS_NODEFILE)
NRANKS_PER_NODE=4
NDEPTH=64

NTOTRANKS=$((NNODES * NRANKS_PER_NODE))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}"
echo <$PBS_NODEFILE

# Initialize environment
module load conda
conda activate base
source ${PBS_O_WORKDIR}/.venv/bin/activate

export TOKENIZERS_PARALLELISM=true

# Logging
echo "$(df -h /local/scratch)"

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=PHB

# For applications that internally handle binding MPI/OpenMP processes to GPUs
mpiexec \
	-n ${NTOTRANKS} \
	--ppn ${NRANKS_PER_NODE} \
	--depth=${NDEPTH} \
	--cpu-bind none \
	--mem-bind none \
	--hostfile $PBS_NODEFILE \
	${PBS_O_WORKDIR}/submit/run-polaris.sh \
	python3 train.py fit --trainer.max_epochs 5 --data.batch_size 128
