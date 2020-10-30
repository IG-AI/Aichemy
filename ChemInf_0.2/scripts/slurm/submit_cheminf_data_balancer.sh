#!/bin/bash

#SBATCH --job-name=cheminf_data_balancer
#SBATCH --time=12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G

#SBATCH -o /home/x_danag/slurm_output/cheminf_data_balancer.%J.out
#SBATCH -e /home/x_danag/slurm_output/cheminf_data_balancer.%J.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=stop,fail

cd /proj/carlssonlab/users/x_danag/ChemInf/

export INPUT_NAME=${INPUT%.*}

module use /proj/carlssonlab/envmod
module load FRONTLINE/latest

echo "-------------------------------------------\n"
echo "START BALANCING OF: "$INPUT_NAME
echo -e "\n-------------------------------------------\n"

srun -N1 --ntasks=1 --cpus-per-task=2 --mem=100G --time=12:00:00 --exclusive \
singularity exec /proj/carlssonlab/singularity/frontline.simg \
python scripts/cheminf_data_utils.py -i $INPUT -o $OUTPUT -m resample -c 100000 -nc $SLURM_CPUS_ON_NODE

# sbatch --export=ALL,INPUT=ChemInf_0.2 cheminf/data/d2.full.csv,OUTPUT=ChemInf_0.2 cheminf/data/d2.full.balanced50-50.csv cheminf/scripts/slurm/submit_cheminf_data_balancer.sh
