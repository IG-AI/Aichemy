#!/bin/bash

#SBATCH --job-name=cheminf_compere_build_rndfor
#SBATCH --time=01:00:00

#SBATCH --ntasks=5
#SBATCH --cpus-per-task=16
#SBATCH --workdir=/proj/carlssonlab/users/x_danag/ChemInf/

#SBATCH -o /home/x_danag/slurm_output/cheminf_compere_build_rndfor.%J.out
#SBATCH -e /home/x_danag/slurm_output/cheminf_compere_build_rndfor.%J.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=none

export NUMBER_MODELS=5
export NUMBER_CLASSES=2
export DATA="ChemInf_0.2-(amcp_0.2_da)/cheminf/data"
export INPUT_TRAIN=$DATA"/d2.trialset.train.6-25.csv"
export INPUT_TEST=$DATA"/d2.trialset.test.6-25.csv"
export PARAMETER_FILE="ChemInf_0.2-(amcp_0.2_da)/cheminf/parameters_d2.txt"

module use /proj/carlssonlab/envmod; module load FRONTLINE/latest

echo "-------------------------------------------"
echo "BUILDING MODELS"
echo "-------------------------------------------"
mkdir $MODEL_DIR
rm -fr $MODEL_DIR"*"
for i in $(seq 0 $(($NUMBER_MODELS - 1)))
do
  echo "Building model: "$i
  srun -N1 --ntasks=1 --cpus-per-task=16 --time=01:00:00 --exclusive \
  singularity exec /proj/carlssonlab/singularity/frontline.simg \
  python $SOURCE -m build -i $INPUT_TRAIN -md $MODEL_DIR"cheminf_model_"$i -p $PARAMETER_FILE &
done

echo "Waiting for models to be built"
wait
echo "Finished building models"

models_directories=$MODEL_DIR"*/"
model_counter=0
for i in $models_directories
do
  echo "Models "$i" been created!"
  model_files=($(ls -1 $i"/" | sed "#.*/##"))
  nr_models_in_dir=$(( ${#model_files[@]} / $(($NUMBER_CLASSES + 1)) ))
  for j in "${model_files[@]}"
  do
    model_number="${j: -3:1}"
    new_m_number=$(($model_number + $model_counter))
    new_file_name="${j%%$model_number.z}$new_m_number.z"
    cp $i/$j $MODEL_DIR$new_file_name
  done
  model_counter=$(($model_counter + $nr_models_in_dir))
  rm $i -r
done

echo "-------------------------------------------"
echo "PREDICTING MODELS"
echo "-------------------------------------------"
split --number=l/$NUMBER_MODELS -d $INPUT_TEST $SOURCE"/chunk_"
chunk_files=$SOURCE"/chunk_*"
chunk_list=($(ls $chunk_files))

for i in $(seq 0 $(( ${#chunk_list[@]} - 1 )))
do
  srun -N1 --ntasks=1 --cpus-per-task=16 --time=01:00:00 --exclusive \
  singularity exec /proj/carlssonlab/singularity/frontline.simg \
  python $SOURCE -m predict -i "${chunk_list[i]}" -md $MODEL_DIR -p $PARAMETER_FILE -o $OUTPUT"_"$i &
done

wait

sub_prediction=$OUTPUT"_*"
tail -n+2 $sub_prediction > $OUTPUT
rm $sub_prediction
rm $chunk_files

#sbatch --export=ALL,MODEL_DIR="cheminf/data/cheminf_models/d2_6-25_models/",SOURCE="cheminf/",OUTPUT="ChemInf_0.2/cheminf/data/cheminf_predictions/d2_predictions/predict_d2_6-25.csv" ChemInf_0.2/scripts/slurm/submit_cheminf_build_predict_rndfor.sh