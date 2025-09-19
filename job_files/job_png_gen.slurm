#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Generate_PNGSe
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --output=Generate_PNG_Out.%j
#SBATCH --partition=cpu
cd $SCRATCH
ml purge 
ml GCC/11.3.0 OpenMPI/4.1.4 TensorFlow/2.13.0-CUDA-11.8.0 matplotlib/3.5.2
source activate_venv acesML
python block-prediction/png_builder.py
