#!/bin/bash

#SBATCH --job-name=cheminf_auto
#SBATCH --time=24:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G

#SBATCH -o /home/x_danag/slurm_output/cheminf_auto.%j.%a.out
#SBATCH -e /home/x_danag/slurm_output/cheminf_auto.%j.%a.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=stop,fail

name=$(awk "NR==${SLURM_ARRAY_TASK_ID}" $NAMES)
config_override=$(awk "NR==${SLURM_ARRAY_TASK_ID}" $CONFIGS)
work_dir=/proj/carlssonlab/users/x_danag/ChemInf/ChemInf_0.2

cd $work_dir

echo -e "--------------------------------------------------------------------------\n"
echo    "STARTING JOB NUMBER "${SLURM_ARRAY_TASK_ID}" WITH NAME "$name
echo -e "\n--------------------------------------------------------------------------\n"

python singularity exec /proj/carlssonlab/singularity/frontline.simg \
python /cheminf auto -cl nn -i $INPUT -n $name -cf $config_override
