#!/bin/bash
#------- qsub option ----------- 
#PBS -q C_dev
#PBS -P GKNET_KE
#PBS -l walltime=00:30:00
#PBS -N python

#PBS -l select=2:mpiprocs=2:ompthreads=1:mem=1510gb
###PBS -l select=2:ncpus=32:mpiprocs=1:ompthreads=1:mem=1510gb

#------- Program execution -----------
module load intel/2025.1
cd $PBS_O_WORKDIR

mpirun python3.12 compress_data.py

