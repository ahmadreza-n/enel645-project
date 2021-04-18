#!/bin/bash

#SBATCH --partition=gpu
#SBATCH -N1
#SBATCH -n3
#SBATCH --gres=gpu:1
#SBATCH --time=1:0:0
#SBATCH --mem=32G

export PATH=/work/TALC/enel645/software/miniconda3/bin:$PATH
source activate tf2gpu

python /home/ahmadreza.nazari/transfer_learning.py