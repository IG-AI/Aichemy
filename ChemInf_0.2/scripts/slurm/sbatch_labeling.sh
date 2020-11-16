#!/bin/bash

#SBATCH --job-name=cheminf_labeling
#SBATCH --time=3:00:00

#SBATCH --nodes=1
#SBATCH --mem=100G

#SBATCH -o /home/x_danag/slurm_output/cheminf_labeling.%j.out
#SBATCH -e /home/x_danag/slurm_output/cheminf_labeling.%j.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=stop,fail

work_dir=/proj/carlssonlab/users/x_danag/ChemInf/ChemInf_0.2

cd $work_dir

full_file=$1

filename=$(echo "${full_file%.*}")

echo -e "Start counting data in file\n"
nlines=$(awk -F"\t" '{print $1}' $full_file | wc -l);
threshold=$(echo "(0.01*$nlines) / 1" | bc)

echo -e "Finding class 1 in file\n"
cut -f1 $full_file | awk -F"_" '$1=$1' OFS="\t" | sort -k2 -n -t$'\t' | \
sed "s/\t/_/" | head -n$threshold > $filename"_class1.csv"

python scripts/labeling.py $filename"_class1.csv" $full_file $filename"_labeled.csv"
