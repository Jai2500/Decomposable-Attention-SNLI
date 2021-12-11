#!/bin/zsh
#SBATCH --time=72:00:00
#SBATCH --mincpus=10
#SBATCH -G 1 -c 10

echo "Activating the virtual environment"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ml

mkdir -p /scratch/wandb2

cd ~/ANLP-Project/

python train.py --gpus=-1 --num_workers=4 --intra_sent_atten=True --val_interval=5000 --use_wandb=True

echo "Done Training"