#!/bin/sh
#SBATCH -c 1
#SBATCH -t 3-12:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
<<<<<<< HEAD
python permutations.py --model_dir exp --steps 60000
=======
python finetune.py --config examples/example1.json
>>>>>>> 93f6fd797e343b9b4301028ba2baf8d707c56c44
