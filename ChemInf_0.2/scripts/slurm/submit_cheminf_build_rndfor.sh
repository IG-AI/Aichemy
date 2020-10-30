#!/bin/bash

#SBATCH --job-name=model_build
#SBATCH --time=01:00:00

#SBATCH --ntasks=5
#SBATCH --cpus-per-task=16
#SBATCH --workdir=/proj/carlssonlab/users/x_danag/ChemInf/

#SBATCH -o /home/x_danag/slurm_output/model_build.%A_%a.out
#SBATCH -e /home/x_danag/slurm_output/model_build.%A_%a.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=all

export NUMBER_MODELS=5
export NUMBER_CLASSES=2
export MODEL_DIR="ChemInf_0.2-(amcp_0.2_da)/cheminf/data/cheminf_models/d2_models/"
export SOURCE="ChemInf_0.2-(amcp_0.2_da)/cheminf/"
export INPUT_TRAIN="ChemInf_0.2-(amcp_0.2_da)/cheminf/data/d2.trialset.train.csv"

module use /proj/carlssonlab/envmod; module load FRONTLINE/latest

echo "-------------------------------------------"
echo "BUILDING MODELS"
echo "-------------------------------------------"
rm -fr $MODEL_DIR"*"
for i in $(seq 0 $(($NUMBER_MODELS - 1)))
do
  echo "Building model: "$i
  srun -N1 --ntasks=1 --cpus-per-task=16 --time=04:00:00 --exclusive \
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
