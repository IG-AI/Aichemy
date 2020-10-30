#!/bin/bash

#SBATCH --job-name=model_validation
#SBATCH --time=01:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=12
#SBATCH --mem=5000M

#SBATCH -o /home/x_danag/slurm_output/model_validation.%J.out
#SBATCH -e /home/x_danag/slurm_output/model_validation.%J.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=end,fail

module use /proj/carlssonlab/envmod; module load FRONTLINE/latest

srun singularity exec /proj/carlssonlab/singularity/frontline.simg python .$HOME/ChemInf/amcp_0.1/amcp/ -m build -i $INPUT_FILE -o ".$HOME/ChemInf/amcp_0.1/amcp/data/valdiation_"INPUT_NAME".csv" -p $HOME/ChemInf/amcp_0.1_da/amcp/source/parameters.txt &
rm -fr $MODEL_DIR"/*"
srun singularity exec /proj/carlssonlab/singularity/frontline.simg python .$HOME/ChemInf/amcp_0.1/amcp/ -m build -i $INPUT_FILE -o ".$HOME/ChemInf/amcp_0.1/amcp/data/valdiation_"$INPUT_NAME_da".csv" -p $HOME/ChemInf/amcp_0.1_da/amcp/source/parameters.txt &
