#!/bin/bash
# Script for running commands within MPICH + Container but before the actual training

# MPI related environmental variables
env | grep PMI
env | grep RANK
env | grep MPI

# Set MPI Variables for Lighting
export WORLD_SIZE=$PMI_SIZE
export RANK=$PMI_RANK
export NODE_RANK=$(($RANK % $PPN))
echo "Rank: $RANK of $WORLD_SIZE. Node: $NODE_RANK of $NUM_OF_NODES."
# Run other commands
$@
