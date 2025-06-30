#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 2-16:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o logs/log_%j.out  
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python finetune.py --model_dir experiments/exp --steps 30000