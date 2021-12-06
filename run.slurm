#!/bin/bash

#SBATCH --job-name=CUSTOM_AUTOKERAS
#SBATCH --error=autom-%j.err
#SBATCH --output=autom-%j.out
#SBATCH --nodes=1
#SBATCH --account=IscrC_ML4COVID
#SBATCH --time=04:00:00
#SBATCH --partition=m100_all_serial
#SBATCH --gres=gpu:1

python autom.py
