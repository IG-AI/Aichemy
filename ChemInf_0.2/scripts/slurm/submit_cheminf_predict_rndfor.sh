#!/bin/sh

#SBATCH --job-name=model_prediction
#SBATCH --time=01:00:00

#SBATCH --ntasks=5
#SBATCH --cpus-per-task=16
#SBATCH --workdir=/proj/carlssonlab/users/x_danag/ChemInf/

#SBATCH -o /home/x_danag/slurm_output/model_prediction.%J.out
#SBATCH -e /home/x_danag/slurm_output/model_prediction.%J.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=all

export NUMBER_MODELS=5
export MODEL_DIR="cheminf_0.1_da/cheminf/data/cheminf_models/d2_models/"
export SOURCE="cheminf_0.1_da/cheminf/"
export INPUT_TEST="cheminf_0.1_da/cheminf/data/d2.trialset.test.csv"
export OUTPUT="cheminf_0.1_da/cheminf/data/cheminf_predictions/d2_predictions/predict_d2.csv"
export PARAMETER_FILE="cheminf_0.1_da/cheminf/parameters_d2.txt"

module use /proj/carlssonlab/envmod; module load FRONTLINE/latest

echo "-------------------------------------------"
echo "PREDICTING MODELS"
echo "-------------------------------------------"
split --number=l/$NUMBER_MODELS -d $INPUT_TEST $SOURCE"/chunk_"
chunk_files=$SOURCE"/chunk_*"
chunk_list=($(ls $chunk_files))

for i in $(seq 0 $(( ${#chunk_list[@]} - 1 )))
do
  srun -N1 --ntasks=1 --cpus-per-task=16 --time=04:00:00 --exclusive \
  singularity exec /proj/carlssonlab/singularity/frontline.simg \
  python $SOURCE -m predict -i "${chunk_list[i]}" -md $MODEL_DIR -p $PARAMETER_FILE -o $OUTPUT"_"$i &
done

wait

sub_prediction=$OUTPUT"_*"
tail -n+2 $sub_prediction > $OUTPUT
rm $sub_prediction
rm $chunk_files