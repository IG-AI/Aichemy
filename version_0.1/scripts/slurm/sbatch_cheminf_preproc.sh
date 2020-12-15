#!/bin/bash

#SBATCH --job-name=cheminf_balancing
#SBATCH --time=12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G

#SBATCH -o /home/x_danag/slurm_output/cheminf_balancing.%j.out
#SBATCH -e /home/x_danag/slurm_output/cheminf_balancing.%j.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=stop,fail

INFILE=d2_full.csv

work_dir=/proj/carlssonlab/users/x_danag/ChemInf/ChemInf_0.2

cd $work_dir

module use /proj/carlssonlab/envmod
module load FRONTLINE/latest

echo -e "--------------------------------------------------------------------------\n"
echo    "STARTING BALANCING "$INFILE
echo -e "\n--------------------------------------------------------------------------\n"

srun -N1 --ntasks=1 --cpus-per-task=2 --mem=100G --time=12:00:00 --exclusive \
singularity exec /proj/carlssonlab/singularity/frontline.simg \
python cheminf/ preproc balancing -i $INFILE -o "cheminf/data/"${INFILE%.*}"_balanced" \
-ch 100000 -nc $SLURM_CPUS_ON_NODE

# sbatch scripts/slurm/sbatch_cheminf_preproc.sh