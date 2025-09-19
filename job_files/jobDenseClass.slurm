#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=DenseClass
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --output=DenseClassOut.%j
#SBATCH --partition=cpu
cd $SCRATCH
ml purge 
ml GCC/12.2.0  OpenMPI/4.1.4 TensorFlow/2.13.0 matplotlib/3.7.0
source activate_venv cpuML
python block-prediction/dense_class_cnn.py
