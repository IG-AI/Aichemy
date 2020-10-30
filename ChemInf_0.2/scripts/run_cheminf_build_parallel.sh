#!/bin/bash

while getopts n:d:s:i:p: flag
do
    case "${flag}" in
        n) number_models=${OPTARG};;
        d) models_dir=${OPTARG};;
        s) input_script=${OPTARG};;
        i) input_file=${OPTARG};;
        p) parameters=${OPTARG};;
    esac
done
export number_models
export models_dir
export input_script
export input_file
export parameters

function parallel_model_build() {
  #echo "python $input_script -m build -i ${file_list[0]} -md ${dir_list[$1]} -p $parameters"
  singularity exec /proj/carlssonlab/singularity/frontline.simg python $input_script -m build -i ${file_list[0]} -md ${dir_list[$1]} -p $parameters
  for i in $(seq 1 $((${#file_list[@]} - 1)));
  do
    #echo "python $input_script -m build -i ${file_list[$i]} -md ${dir_list[$1]} -p $parameters"
    singularity exec /proj/carlssonlab/singularity/frontline.simg python $input_script -m improve -i ${file_list[$i]}  -md ${dir_list[$1]} -p $parameters
  done
}

function ceil() {                                                                       
  echo "define ceil (x) {if (x<0) {return x/1} \
        else {if (scale(x)==0) {return x} \
        else {return x/1 + 1 }}} ; ceil($1)" | bc
}

range=$((number_models - 1));
for i in $(seq 0 $range);
do
  cheminf_model=$models_dir"cheminf_models_"$i
  dir_list[$i]="$cheminf_model"
  mkdir "$cheminf_model"
done
export dir_list

{
  {
    module use /proj/carlssonlab/envmod; module load FRONTLINE/latest
    } && {
    if [ "$SLURM_MEM_PER_CPU" -lt "$number_models" ]
    then
      return
    fi

    if [[ -L "$input_file" ]]
    then
      input_file=$(readlink $input_file)
    fi

    input_file_size=$(( $( stat -c '%s' $input_file ) / 1024 / 1024 ))
    file_split_float=$(echo "$input_file_size / (($SLURM_MEM_PER_CPU * $SLURM_JOB_CPUS_PER_NODE) / $number_models)" | bc -l)
    number_chunk=$(($(ceil $file_split_float) + 2));
    if [ "$number_chunk" -gt "1" ]
    then
      echo "Creating "$number_chunk" chunk(s) from input "$input_file
      total_lines=$(wc -l $input_file | awk '{print $1; exit}')
      ((lines_per_file = (total_lines + number_chunk - 1) / number_chunk))
      split --number=l/$number_models -d $input_file $models_dir"chunk_"
      file_list=($(ls -f -p "$models_dir" | grep -v /))
      for i in $(seq 0 $range);
      do
        file_list[i]=$models_dir${file_list[i]}
      done
    else
      file_list[0]=$input_file
    fi

    parallel_model_build 0 &
    for i in $(seq 0 $(($number_models - 1)));
    do
      echo "Starting model training for model: "$i
      parallel_model_build $i > /dev/null 2>&1 &
    done
  }
} || {
  python $input_script -m build -i $input_file -md ${dir_list[$(($number_models - 1))]} -p $parameters &
  for i in $(seq 1 $(($range - 1)));
  do
    nohup python $input_script -m build -i $input_file -md ${dir_list[$i]} -p $parameters \
    > /dev/null 2>&1 &
  done
}

wait < <(jobs -p)
test -f fail && echo "The models didnt finish the training."

if [ "${#file_list[@]}" -gt "1" ]
then
for i in "${file_list[@]}"

  do
    rm $i
  done
fi

models_directories=$models_dir"*/"
model_counter=0

for i in $models_directories
do
  model_files=($(ls -1 $i"/" | sed "#.*/##"))
  nr_models_in_dir=$((${#model_files[@]} / 3))
  for j in "${model_files[@]}"
  do
    model_number="${j: -3:1}"
    new_m_number=$(($model_number + $model_counter))
    new_file_name="${j%%$model_number.z}$new_m_number.z"
    cp $i/$j $models_dir$new_file_name
  done
  model_counter=$(($model_counter + $nr_models_in_dir))
  rm $i -r
done
