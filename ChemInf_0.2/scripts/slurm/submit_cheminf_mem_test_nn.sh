#!/bin/bash

#SBATCH --job-name=cheminf_mem_test_nn
#SBATCH --time=12:00:00

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10G
#SBATCH --workdir=/proj/carlssonlab/users/x_danag/ChemInf

#SBATCH -o /home/x_danag/slurm_output/cheminf_mem_test_nn.%A.out
#SBATCH -e /home/x_danag/slurm_output/cheminf_mem_test_nn.%A.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=all

#SBATCH --array=0-8

export INPUT_TRAIN="cheminf/data/d2.trialset.train.csv"
export INPUT_TEST="cheminf/data/d2.trialset.test.csv"
export DIR="NC_210_v2/data/mem_test/d2_original"

module use /proj/carlssonlab/envmod; module load FRONTLINE/latest

sub_dir=$DIR"/d2_dir_"$SLURM_ARRAY_TASK_ID
input_train_name=$(basename "${INPUT_TRAIN%.*}")
input_test_name=$(basename "${INPUT_TEST%.*}")
new_train=$sub_dir"/"$input_train_name"_cat.csv"
new_test=$sub_dir"/"$input_test_name"_cat.csv"
file_div=$(printf %.6f "$((10**6 * 1/(2**$SLURM_ARRAY_TASK_ID)))e-6")

echo $new_train
echo $new_test

mkdir $sub_dir

singularity exec /proj/carlssonlab/singularity/frontline.simg \
python scripts/cheminf_split_file.py \
-i $INPUT_TRAIN -o $new_train -m cat -p $file_div

singularity exec /proj/carlssonlab/singularity/frontline.simg \
python scripts/cheminf_split_file.py \
-i $INPUT_TEST -o $new_test -m cat -p $file_div

new_train_head=$new_train"_head"
head -n1 NC_210_v2/data/header.txt \
| cat - $new_train | tr [:blank:] \\t \
> $new_train_head

new_test_head=$new_test"_head"
head -n1 NC_210_v2/data/header.txt \
| cat - $new_test | tr [:blank:] \\t \
> $new_test_head

rm $new_test
rm $new_train

singularity exec /proj/carlssonlab/singularity/frontline.simg \
python "NC_210_v2/"01_DNN_CP_tr_te_20_original.py -i $new_train_head -f fp -al 4layer -p $new_test_head
