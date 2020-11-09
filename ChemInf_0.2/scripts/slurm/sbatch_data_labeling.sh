#!/bin/bash

#SBATCH --job-name=cheminf_data_labeling
#SBATCH --time=12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G

#SBATCH -o /home/x_danag/slurm_output/cheminf_data_labeling.%j.out
#SBATCH -e /home/x_danag/slurm_output/cheminf_data_labeling.%j.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=stop,fail

INFILE=cheminf/data/d2_full.csv

work_dir=/proj/carlssonlab/users/x_danag/ChemInf/ChemInf_0.2

cd $work_dir

filename=$(echo "${INFILE%.*}")

echo -e "-------------------------------------------\n"
echo "START LABELING OF: "$INFILE
echo -e "\n-------------------------------------------\n"

echo -e "Start counting data in file\n"
nlines=$(awk -F "\t" '{print $1}' $INFILE | wc -l);
threshold=$(echo "(0.01*$nlines) / 1" | bc)

echo -e "Finding class 1 in file\n"
cut -f1 $INFILE | awk -F"_" '$1=$1' OFS="\t" | sort -k2 -n -t$'\t' | \
sed "s/\t/_/" | head -n$threshold > $filename"_class1.csv"

echo -e "Labeling data in file\n"
while read line;
do
  if grep -q "$(echo $line | awk -F" " '{print $1}')" $filename"_class1.csv";
  then
    echo $line | sed "s/ / 1 /" | tr ' ' '\t' >> $filename"_labeled.csv";
  else
    echo $line | sed "s/ / 0 /" | tr ' ' '\t' >> $filename"_labeled.csv";
  fi
done <$INFILE