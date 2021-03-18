#! /bin/bash
#SBATCH --job-name=transfer-segmentations
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time="48:00:00"
#SBATCH --output='slurm_logs/segmentation_transfer_%A_%a.out'
#SBATCH --error='slurm_logs/segmentation_transfer_%A_%a.out'
#SBATCH --partition compute
#SBATCH --array 0-19
#SBATCH --cpus-per-task=10
#SBATCH --mem=100gb
PATH=$PATH:/usr/bin/git
export PATH


GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git
export GIT_PYTHON_GIT_EXECUTABLE
GIT_PYTHON_REFRESH=quiet

echo "$SLURM_ARRAY_TASK_ID th decile"
start_frac=$(echo "scale=2 ; $SLURM_ARRAY_TASK_ID / 20" | bc)
end_frac=$(echo "scale=2 ; ($SLURM_ARRAY_TASK_ID+1) / 20" | bc)
echo $start_frac
echo $end_frac
python transfer_all_segmentations.py --start_frac=$start_frac --end_frac=$end_frac --set_type=train
python transfer_all_segmentations.py --start_frac=$start_frac --end_frac=$end_frac --set_type=val
python transfer_all_segmentations.py --start_frac=$start_frac --end_frac=$end_frac --set_type=test
