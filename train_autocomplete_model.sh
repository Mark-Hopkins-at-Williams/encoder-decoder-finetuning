#!/bin/sh
#SBATCH -c 1
#SBATCH -t 5-12:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python autocomplete.py