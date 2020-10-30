#!/bin/bash

#SBATCH --job-name=cheminf_summaries_rndfor
#SBATCH --time=00:10:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --workdir=/proj/carlssonlab/users/x_danag/ChemInf/

#SBATCH -o /home/x_danag/slurm_output/cheminf_summaries_rndfor.%J.out
#SBATCH -e /home/x_danag/slurm_output/cheminf_summaries_rndfor.%J.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=none

export INPUT_NAME=${INPUT%.*}

module use /proj/carlssonlab/envmod; module load FRONTLINE/latest

echo "-------------------------------------------"
echo "CREATE SUMMARY OF PREDICTION: "$INPUT_NAME
echo "-------------------------------------------"

srun -N1 --ntasks=1 --cpus-per-task=1 --time=00:10:00 --exclusive \
singularity exec /proj/carlssonlab/singularity/frontline.simg \
python "scripts/cheminf_summarize_pred.py" \
-i $INPUT -o ${INPUT%.*}"_sum.csv"

srun -N1 --ntasks=1 --cpus-per-task=1 --time=00:10:00 --exclusive \
singularity exec /proj/carlssonlab/singularity/frontline.simg \
python "scripts/cheminf_plot_summary.py" \
-i ${INPUT%.*}"_sum.csv" -p $INPUT_NAME

#sbatch --export=ALL,INPUT= scripts/slurm/submit_cheminf_summaries_nn.sh