#!/bin/bash

#SBATCH --job-name=noise_comp_cheminf
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G

#SBATCH -o /home/x_danag/slurm_output/%x.%A.%a.out
#SBATCH -e /home/x_danag/slurm_output/%x.%A.%a.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=none

#SBATCH --array=1-10

work_dir=/proj/carlssonlab/users/x_danag/ChemInf/ChemInf_0.2

cd $work_dir

name=$(awk NR=="$SLURM_ARRAY_TASK_ID" "$SETUP" | cut -d$'\t' -f1)
config_override=$(awk NR=="$SLURM_ARRAY_TASK_ID" "$SETUP" | cut -d$'\t' -f2)
input=$(awk NR=="$SLURM_ARRAY_TASK_ID" "$SETUP"| cut -d$'\t' -f3)

extension="${input##*.}"
filename="${input%.*}"
percentage=$(echo "${input%.*}" | grep -Eo "0.+(\.[0-9]+)?")

module use /proj/carlssonlab/envmod; module load FRONTLINE/latest

echo -e "--------------------------------------------------------------------------\n"
echo    "STARTING JOB NUMBER "$SLURM_ARRAY_TASK_ID" WITH NAME "$name
echo -e "\n--------------------------------------------------------------------------\n"

srun -N1 --ntasks=1 --cpus-per-task=4 --mem=300G --time=24:00:00 --exclusive \
singularity exec /proj/carlssonlab/singularity/frontline.simg \
python cheminf/ build -cl nn -i "cheminf/data/"$name"/"$filename"_train."$extension -n $name -cf $config_override

srun -N1 --ntasks=1 --cpus-per-task=4 --mem=300G --time=24:00:00 --exclusive \
singularity exec /proj/carlssonlab/singularity/frontline.simg \
python cheminf/ predict -cl nn -i "cheminf/data/"$name"/"$filename"_test."$extension  -n $name -cf $config_override

srun -N1 --ntasks=1 --cpus-per-task=1 --mem=300G --time=24:00:00 --exclusive \
singularity exec /proj/carlssonlab/singularity/frontline.simg \
python cheminf/ postproc summary -i "cheminf/data/"$name"/predictions/"$name"_nn_predictions.csv" \
-n $name -cf $config_override

srun -N1 --ntasks=1 --cpus-per-task=1 --mem=300G --time=24:00:00 --exclusive \
singularity exec /proj/carlssonlab/singularity/frontline.simg \
python cheminf/ postproc plot -i "cheminf/data/"$name"/predictions/"$name"_nn_predictions_summary.csv" \
-n $name -cf $config_override

# sbatch --export=ALL,SETUP=scripts/slurm/projects/complement_noise.csv scripts/slurm/sbatch_cheminf_build_predict_noise.sh