#!/bin/bash
#
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00
#SBATCH --mem=1G
 
export M5_PATH=/data/gem52k
export LAB_PATH=$PWD
export GEM5_BIN=$LAB_PATH/build/X86/gem5.opt
 
srun run_all.sh