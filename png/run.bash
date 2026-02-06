#!/bin/bash

time python png_builder2.py --size 2000 --name dataset_2000 --sample 5000 --save_raw --skip-gmres --image-size 128
time python png_builder2.py --size 5000 --name dataset_5000 --sample 5000 --save_raw --skip-gmres --image-size 128
#time python png_builder2.py --size 1000 --name dataset_1000_script --sample 5000 --save_raw
#time python png_builder2.py --size 2500 --name png_dataset_2500_script --sample 5000
#time python png_builder2.py --size 5000 --name png_dataset_5000_script --sample 5000

