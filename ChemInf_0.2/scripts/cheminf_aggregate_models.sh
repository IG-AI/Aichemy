#!/bin/bash

# This script is used to make an aggregated directory of classifiers along
#  with their calibration conformity files in a subsequent to cheminf-build.
#  Use this script in the directory where the subdirectories contain the
#  generated amcp_models.

rm -r amcp_models_agg
model_counter=0
models_directories=($(ls -d */))
mkdir amcp_models_agg

for i in "${models_directories[@]}"
do
	model_files=($(ls $i))
	nr_models_in_dir=$((${#model_files[@]} / 3))
	for j in "${model_files[@]}"
	do
		model_number="${j: -3:1}"
		new_m_number=$(($model_number + $model_counter))
		new_file_name="${j%%$model_number.z}$new_m_number.z"
		cp $i/$j amcp_models_agg/$new_file_name
	done
	model_counter=$(($model_counter + $nr_models_in_dir))
done

mv amcp_models_agg/* .
rm amcp_models_* -r
