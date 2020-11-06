#!/bin/bash

#SBATCH --job-name=cheminf_auto
#SBATCH --time=24:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G

#SBATCH -o /home/x_danag/slurm_output/cheminf_auto.%j.%a.out
#SBATCH -e /home/x_danag/slurm_output/cheminf_auto.%j.%a.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=stop,fail

#SBATCH --array=1-14

SETUP=scripts/slurm/projects/inv_nn_different_sampling_setup.csv
INFILE=cheminf/data/d2_full.csv

work_dir=/proj/carlssonlab/users/x_danag/ChemInf/ChemInf_0.2

cd $work_dir

name=$(awk NR=="$SLURM_ARRAY_TASK_ID" "$SETUP" | cut -d$'\t' -f1)
config_override=$(awk NR=="$SLURM_ARRAY_TASK_ID" "$SETUP" | cut -d$'\t' -f2)

module use /proj/carlssonlab/envmod; module load FRONTLINE/latest

echo -e "--------------------------------------------------------------------------\n"
echo    "STARTING JOB NUMBER "$SLURM_ARRAY_TASK_ID" WITH NAME "$name
echo -e "\n--------------------------------------------------------------------------\n"

singularity exec /proj/carlssonlab/singularity/frontline.simg \
python cheminf/ auto -cl nn -i $INFILE -n $name -cf $config_override

# sbatch scripts/slurm/sbatch_cheminf_auto.sh