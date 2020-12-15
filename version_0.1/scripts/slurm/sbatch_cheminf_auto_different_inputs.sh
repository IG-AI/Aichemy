#!/bin/bash

#SBATCH --job-name=conf_noiz_cheminf
#SBATCH --time=36:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G

#SBATCH -o /home/x_danag/slurm_output/%x.%A.%a.out
#SBATCH -e /home/x_danag/slurm_output/%x.%A.%a.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=none

#SBATCH --array=1-35

work_dir=/proj/carlssonlab/users/x_danag/ChemInf/ChemInf_0.2

cd $work_dir

name=$(awk NR=="$SLURM_ARRAY_TASK_ID" "$SETUP" | cut -d$'\t' -f1)
config_override=$(awk NR=="$SLURM_ARRAY_TASK_ID" "$SETUP" | cut -d$'\t' -f2)
input=$(awk NR=="$SLURM_ARRAY_TASK_ID" "$SETUP"| cut -d$'\t' -f3)

module use /proj/carlssonlab/envmod; module load FRONTLINE/latest

echo -e "--------------------------------------------------------------------------\n"
echo    "STARTING JOB NUMBER "$SLURM_ARRAY_TASK_ID" WITH NAME "$name
echo -e "\n--------------------------------------------------------------------------\n"

srun -N1 --ntasks=1 --cpus-per-task=4 --mem=100G --time=36:00:00 --exclusive \
singularity exec /proj/carlssonlab/singularity/frontline.simg \
python cheminf/ auto -i $input -cl nn -n $name -cf $config_override -ch 100000 -nc 4

# sbatch --export=ALL,SETUP=scripts/slurm/projects/inv_nn_noiselevels_setup.csv scripts/slurm/sbatch_cheminf_auto_different_inputs.sh
