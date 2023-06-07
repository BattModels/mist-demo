#!/bin/sh
env
export WORLD_SIZE=$PMI_SIZE
export RANK=$PMI_RANK
export CUDA_LAUNCH_BLOCKING=1
exec "$@"
