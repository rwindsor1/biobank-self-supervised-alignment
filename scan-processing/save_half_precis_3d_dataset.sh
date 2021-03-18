#! /bin/bash
#SBATCH --job-name=make3dDataset
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time="48:00:00"
#SBATCH --output='slurm_logs/save_half_precis_3d_dataset/%A_%a.out'
#SBATCH --error='slurm_logs/save_half_precis_3d_dataset/%A_%a.out'
#SBATCH --partition compute
#SBATCH --array 0-19
#SBATCH --cpus-per-task=25
#SBATCH --mem=100gb
echo "$SLURM_ARRAY_TASK_ID th decile"
python save_half_precis_3d_dataset.py --frac $SLURM_ARRAY_TASK_ID
