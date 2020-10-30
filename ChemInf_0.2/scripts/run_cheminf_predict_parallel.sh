#!/bin/bash

while getopts l:d:s:i:o:t: flag
do
    case "${flag}" in
        l) number_compunds=${OPTARG};;
        d) models_dir=${OPTARG};;
        s) input_script=${OPTARG};;
        i) input_file=${OPTARG};;
        o) output_file=${OPTARG};;
        t) temp_dir=${OPTARG};;
    esac
done

mkdir "$temp_dir"
split_name=$input_file".split_"
temp_dir_atrix=$temp_dir"/*"
$split_name -d -a 3 -l $number_compunds --additional-suffix= $temp_dir
echo $temp_dir_atrix
file_list=$(ls $temp_dir_atrix)
echo $file_list
number_files=${#file_list[@]}
echo $number_files

{
  {
    module use /proj/carlssonlab/envmod; module load FRONTLINE/latest
    } && {
    for i in $(seq 0 $(($number_files - 2)));
    do
      nohup singularity exec /proj/carlssonlab/singularity/frontline.simg \
      python $input_script -m build -i file_list[i] $ -md "${dir_list[$i]}" -p $parameters  \
      > /dev/null 2>&1 &
    done
    singularity exec /proj/carlssonlab/singularity/frontline.simg \
    python $input_script -m predict -i $file_list[$(($number_files - 1))] -md $models_dir -o $output
  }
} || {
  for i in $(seq 0 $(($range - 1)));
  do
    nohup python $input_script -m build -i file_list[i] $ -md "${dir_list[$i]}" -p $parameters > /dev/null 2>&1 &
  done
  python python $input_script -m predict -i $file_list[$(($number_files - 1))] -md $models_dir -o $output
}

rm -fr temp_dir