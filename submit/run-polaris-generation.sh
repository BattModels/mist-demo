#!/bin/sh
# Pulled from: https://github.com/ramanathanlab/genslm/blob/main/genslm/hpc/templates/run-polaris-generation.sh
# MIT License
# Copyright (c) 2022 ramanathanlab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

export NRANKS=$(wc -l <"${PBS_NODEFILE}")
export NODE_RANK=$(($PMI_RANK % $NRANKS))
export SUBNODE_RANK=$(($NODE_RANK % 4)) # get gpu device rank
echo "NODE_RANK: $NODE_RANK, SUBNODE_RANK: $SUBNODE_RANK, LOCAL_RANK: $OMPI_COMM_WORLD_LOCAL_RANK"
export CUDA_LAUNCH_BLOCKING=1
exec "$@"
