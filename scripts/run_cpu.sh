#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 2-16:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o ../logs/log_%j.out  
#SBATCH -e ../logs/log_%j.err
srun -u python extract_vocab.py --model_dir exp --steps 60000