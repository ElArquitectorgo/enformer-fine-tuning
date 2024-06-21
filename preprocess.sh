#!/bin/bash

#SBATCH --job-name=preprocess
#SBATCH --time=0-30:0
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
##SBATCH --ntasks-per-node=2
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:2

#SBATCH --output=preprocess_mouse_%A-rna.out    # Standard output and error log
#SBATCH --error=preprocess_mouse_%A-rna.err    # Standard output and error log

date
hostname
#export SLURM_ARRAYID
##echo SLURM_ARRAYID: $SLURM_ARRAYID
#echo TASKID: $SLURM_ARRAY_TASK_ID

##time python bin/basenji_data.py -s .9 -g data/hg38_gaps.bed -b data/hg38.blacklist.rep.bed -l 196608 --local -o data/human -p 128 -v .1 -w 128 data/genome.fa data/rna_data_filtered.txt --restart
time python bin/basenji_data.py -s .9 -g data/mm10_gaps.bed -b data/mm10.blacklist.rep.bed -l 196608 --local -o data/mouse -p 128 -v .1 -w 128 data/mm10.fa data/targets_mouse.txt

cut -f4 data/human/sequences.bed | sort | uniq -c
head -n3 data/human/sequences.bed
grep valid data/human/sequences.bed | head -n3
grep test data/human/sequences.bed | head -n3
ls -l data/human/tfrecords/*.tfr