#!/bin/bash -l
#SBATCH -J test # Job name
#SBATCH --time=50:00          # Runtime in D-HH:MM
#SBATCH -A che210007p              # Partition to submit to #gpu
#SBATCH -p GPU    # RM-shared
#SBATCH --nodes=2                # node count
#SBATCH --gres=gpu:v100-32:8
#SBATCH --ntasks-per-node=8      # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --mail-type=ALL         # Type of email notification- BEGIN,END,ALL,FAIL
#SBATCH --mail-user=anoushkb@andrew.cmu.edu

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

module load anaconda3
module load cuda/11.7
# activate conda env
source activate incite

export PL_TORCH_DISTRIBUTED_BACKEND=nccl

nvidia-smi --list-gpus
srun python /ocean/projects/che210007p/abhutani/electolyte_fm/electrolyte_fm/training/mlm_pretraining.py