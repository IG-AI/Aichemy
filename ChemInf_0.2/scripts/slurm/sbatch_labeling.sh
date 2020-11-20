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
suffix=$3
percentage=$2

filename=$(echo "${full_file%.*}")

echo -e "--------------------------------------------------------------------------\n"
echo    "STARTING LABELING OF "$name
echo -e "\n--------------------------------------------------------------------------\n"

echo -e "Start counting data in file\n"
nlines=$(awk -F"\t" '{print $1}' $full_file | wc -l);

echo -e "Finding threshold position\n"
threshold=$(echo "($percentage*$nlines) / 1" | bc)

echo -e "Extracting class 1 data in file\n"
cut -f1 $full_file | awk -F"_" '$1=$1' OFS="\t" | sort -k2 -n -t$'\t' | \
sed "s/\t/_/" | head -n$threshold > $filename"_class1"$suffix".csv"

echo -e "Start labeling data\n"
python scripts/labeling.py $filename"_class1.csv" $full_file $filename"_"$suffix".csv"
