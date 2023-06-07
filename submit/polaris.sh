#!/bin/bash
#PBS -l place=scatter
#PBS -l select=1:system=polaris
#PBS -l walltime=0:10:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -q debug
#PBS -A fm_electrolyte
set -x
cd ${PBS_O_WORKDIR}

source "${PBS_O_WORKDIR}/submit/proxy_settings.sh"

module load singularity
module load cray-mpich-abi
ADDITIONAL_PATH=/opt/cray/pe/pals/1.1.7/lib/
export SINGULARITYENV_LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:$ADDITIONAL_PATH"

# Set ADDR and PORT for communication
master_node=$(cat $PBS_NODEFILE | head -1)
export MASTER_ADDR=$(host $master_node | head -1 | awk '{print $4}')
export MASTER_PORT=2345

# NCCL settings
export NCCL_DEBUG=info
export NCCL_NET_GDR_LEVEL=PHB

# Setup workld
export NUM_OF_NODES=$(wc -l <$PBS_NODEFILE)
PPN=4
PROCS=$(($NUM_OF_NODES * PPN))
echo "NUM_OF_NODES= ${NODES} TOTAL_NUM_RANKS= ${PROCS} RANKS_PER_NODE= ${PPN}"

# Actually launch the job
# - Need to `--no-home` to avoid issue with mpich + other MPIs
# - Need to bind in the working directory to have access to the code
mpiexec \
	-hostfile "$PBS_NODEFILE" \
	-n $PROCS \
	-ppn $PPN \
	singularity exec \
	-B /opt \
	-B /var/run/palsd/ \
	-B "${PBS_O_WORKDIR}":/fme \
	--nv \
	"${PBS_O_WORKDIR}/containers/polaris-deep-learning.sif" \
	bash "/fme/submit/polaris_launcher.sh" \
	python /fme/train.py fit --trainer.max_epochs=4
