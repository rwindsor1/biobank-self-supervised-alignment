#! /bin/bash
#SBATCH --job-name=stitch_scans
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time="48:00:00"
#SBATCH --output='slurm_logs/slurm_jobs_out/stitch_decile_%A_%a.out'
#SBATCH --error='slurm_logs/slurm_jobs_error/stitch_decile_%A_%a.out'
#SBATCH --partition compute
#SBATCH --array 0-19
#SBATCH --cpus-per-task=10
#SBATCH --mem=10gb
echo "$SLURM_ARRAY_TASK_ID th decile"
start_frac=$(echo "scale=2 ; $SLURM_ARRAY_TASK_ID / 20" | bc)
end_frac=$(echo "scale=2 ; ($SLURM_ARRAY_TASK_ID+1) / 20" | bc)
echo $start_frac
echo $end_frac
python make_synthesize_coronal_slices.py --start_frac=$start_frac --end_frac=$end_frac --set_type=train
python make_synthesize_coronal_slices.py --start_frac=$start_frac --end_frac=$end_frac --set_type=val
python make_synthesize_coronal_slices.py --start_frac=$start_frac --end_frac=$end_frac --set_type=test
