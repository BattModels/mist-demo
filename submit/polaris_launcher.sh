#!/bin/bash
# Script for running commands within MPICH + Container but before the actual training

# MPI related environmental variables
env | grep PMI
env | grep RANK
env | grep MPI

# Everything else
env

# Run other commands
$@
