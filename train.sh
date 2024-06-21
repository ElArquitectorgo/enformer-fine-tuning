#!/bin/bash

#SBATCH --job-name=train
#SBATCH --time=0-100:0
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
##SBATCH --ntasks-per-node=1
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

#SBATCH --output=Adam_400_both_%A.out    # Standard output and error log
#SBATCH --error=Adam_400_both_%A.err    # Standard output and error log

date
hostname
#export SLURM_ARRAYID
##echo SLURM_ARRAYID: $SLURM_ARRAYID
#echo TASKID: $SLURM_ARRAY_TASK_ID

time python enformer/train_both.py