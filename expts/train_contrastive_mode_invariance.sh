#! /bin/bash 
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time="48:00:00"
#SBATCH --error="slurm_job_array_output/train_contrastive_%A_%a.out"
#SBATCH --output="slurm_job_array_output/train_contrastive_%A_%a.out"
#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=100gb
cd /users/rhydian/self-supervised-project
source /users/rhydian/.bash_profile
conda activate torch_env
bash copy_all_to_temp.sh
python train_contrastive_mode_invariance.py with BATCH_SIZE=10 MARGIN=1 NOTE='WiderMargin' LR=0.00001


