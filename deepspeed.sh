#!/bin/bash -l
# The first 15 characters of the job name are displayed in the qstat output:
#PBS -N deepspeed
# -------------------------------------------------------------------------------------------------------------------
# To submit this script on Polaris:
# qsub -A <PROJECT> -V -q debug-scaling -l select=2 -l walltime=01:00:00 -l filesystems=home:grand:eagle deepspeed.sh
# -------------------------------------------------------------------------------------------------------------------
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

module load conda/2022-07-19
conda activate base
source $PBS_O_WORKDIR/.venv/bin/activate
echo python3: $(which python3)

source "${PBS_O_WORKDIR}/submit/proxy_setting.sh"

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job ID: ${PBS_JOBID}"
echo "Job started at: ${TSTAMP}"

export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB

NRANKS=$(wc -l <"${PBS_NODEFILE}")
NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
NGPUS="$((${NRANKS} * ${NGPU_PER_RANK}))"
echo "NRANKS: ${NRANKS}, NGPU_PER_RANK: ${NGPU_PER_RANK}, NGPUS: ${NGPUS}"

mpiexec \
	--verbose \
	--envall \
	-n "${NGPUS}" \
	--ppn "${NGPU_PER_HOST}" \
	--hostfile="${PBS_NODEFILE}" \
	--cpu-bind verbose,list:0,8,16,24 \
	python3 \
	train.py fit --trainer.fast_dev_run 100
