#!/bin/bash

#SBATCH --job-name=cheminf_balancing
#SBATCH --time=24:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G

#SBATCH -o /home/x_danag/slurm_output/cheminf_balancing.%j.out
#SBATCH -e /home/x_danag/slurm_output/cheminf_balancing.%j.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=stop,fail

work_dir=/proj/carlssonlab/users/x_danag/ChemInf/ChemInf_0.2

cd $work_dir

echo -e "--------------------------------------------------------------------------\n"
echo    "STARTING BALANCING "$INFILE
echo -e "\n--------------------------------------------------------------------------\n"

python singularity exec /proj/carlssonlab/singularity/frontline.simg \
python /cheminf preproc balancing -i $INFILE -o "cheminf/data/"${INFILE%.*}"_balanced" -ch 100000 -nc 2
