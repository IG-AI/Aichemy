#!/bin/bash

#SBATCH --job-name=arch_inv_cheminf
#SBATCH --time=48:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G

#SBATCH -o /home/x_danag/slurm_output/%x.%A.%a.out
#SBATCH -e /home/x_danag/slurm_output/%x.%A.%a.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=none

#SBATCH --array=1-35

work_dir=/proj/carlssonlab/users/x_danag/ChemInf/ChemInf_0.2

cd $work_dir

name=$(awk NR=="$SLURM_ARRAY_TASK_ID" "$SETUP" | cut -d$'\t' -f1)
config_override=$(awk NR=="$SLURM_ARRAY_TASK_ID" "$SETUP" | cut -d$'\t' -f2)

module use /proj/carlssonlab/envmod; module load FRONTLINE/latest

echo -e "--------------------------------------------------------------------------\n"
echo    "STARTING JOB NUMBER "$SLURM_ARRAY_TASK_ID" WITH NAME "$name
echo -e "\n--------------------------------------------------------------------------\n"

srun -N1 --ntasks=1 --cpus-per-task=4 --mem=300G --time=48:00:00 --exclusive \
singularity exec /proj/carlssonlab/singularity/frontline.simg \
python cheminf/ auto -cl nn -i $INFILE -n $name -cf $config_override -ch 100000 -nc 4

# sbatch --export=ALL,INFILE=cheminf/data/d2_full.csv,SETUP=scripts/slurm/projects/inv_nn_balancing_ratio.csv scripts/slurm/sbatch_cheminf_auto.sh