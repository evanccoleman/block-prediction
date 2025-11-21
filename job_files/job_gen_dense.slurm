#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Generate_Dense
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=Gen_Dense.%j
#SBATCH --partition=cpu
cd $SCRATCH
ml purge 
ml GCC/11.3.0 OpenMPI/4.1.4 TensorFlow/2.13.0-CUDA-11.8.0 matplotlib/3.5.2
source activate_venv acesML
python block-prediction/matrix_builder.py
