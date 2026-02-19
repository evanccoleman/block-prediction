#!/bin/bash

date
time ./run_full_comparison.sh ../../dataset_128 50 1>>output_128_1 2>>output_128_2
time ./run_full_comparison.sh ../../dataset_500 50 1>>output_500_1 2>>output_500_2
time ./run_full_comparison.sh ../../dataset_1000 50 1>>output_1000_1 2>>output_1000_2
time ./run_full_comparison.sh ../../dataset_2500 50 1>>output_2500_1 2>>output_2500_2
time ./run_full_comparison.sh ../../dataset_5000 50 1>>output_5000_1 2>>output_5000_2
time ./run_full_comparison.sh ../../dataset_10000 50 1>>output_10000_1 2>>output_10000_2
time ./run_transfer_study.sh ../../dataset_128 ../../dataset_500 ../../dataset_1000 ../../dataset_2500 ../../dataset_5000 ../../dataset_10000 1>>output_transfer_1 2>>output_transfer_2
date
