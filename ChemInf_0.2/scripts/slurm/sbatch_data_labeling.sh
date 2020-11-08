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

echo -e "Start sorting data in file\n"
awk -F"_" '$1=$1' OFS="\t" $INFILE | sort -k2 -n -t$'\t' | sed "s/\t/_/" > $filename"_sorted.csv"

echo -e "Start counting data in file\n"
nlines=$(awk -F "\t" '{print $1}' $INFILE | wc -l);
threshold=$(echo "(0.01*$nlines) / 1" | bc)

echo -e "Extracting class 1\n"
head -n $threshold  $filename"_sorted.csv" | sed "s/\t/$(echo -e '\t')1$(echo -e '\t')/" | \
 awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*$threshold, $0;}' | sort -n | cut -c8- \
> $filename"_class1.csv"

echo -e "Extracting class 0\n"
tail -n -$threshold  $filename"_sorted.csv" | sed "s/\t/$(echo -e '\t')0$(echo -e '\t')/" | \
awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*$(($nlines - $threshold)), $0;}' | sort -n | cut -c8- \
> $filename"_class0.csv"

cat $filename"_class1.csv" < $filename"_class0.csv" | \
awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*$((2 * $threshold)), $0;}' | sort -n | cut -c8- \
> $filename"_labeled.csv"