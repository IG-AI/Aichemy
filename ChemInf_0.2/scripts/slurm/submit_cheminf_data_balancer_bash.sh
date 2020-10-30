#!/bin/bash

#SBATCH --job-name=cheminf_data_balancer_bash
#SBATCH --time=01:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH -o /home/x_danag/slurm_output/cheminf_data_balancer_bash.%J.out
#SBATCH -e /home/x_danag/slurm_output/cheminf_data_balancer_bash.%J.err
#SBATCH --mail-user daniel.agstrand.5971@student.uu.se
#SBATCH --mail-type=stop,fail

cd /proj/carlssonlab/users/x_danag/ChemInf/

filename=$(echo "${INPUT%.*}")

echo -e "-------------------------------------------\n"
echo "START BALANCING OF: "$INPUT
echo -e "\n-------------------------------------------\n"

echo -e "Start counting data in file\n"
nlines=$(awk -F " " '{print $1}' $INPUT | wc -l);
threshold=$(echo "(0.01*$nlines) / 1" | bc)

echo -e "Extracting class 1\n"
head -n $threshold  $INPUT | awk -F"_" '$1=$1' OFS=" " | sort -k2 -n -t" " | sed "s/ /_/" | \
sed "s/\t/$(echo -e '\t')1$(echo -e '\t')/" | \
awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*$threshold, $0;}' | sort -n | cut -c8- \
> $filename".class1.csv"

echo -e "Extracting class 0\n"
tail -n -$threshold  $INPUT | awk -F"_" '$1=$1' OFS=" " | sort -k2 -n -t" " | sed "s/ /_/" | \
sed "s/\t/$(echo -e '\t')0$(echo -e '\t')/" | \
awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*$(($nlines - $threshold)), $0;}' | sort -n | cut -c8- \
> $filename".class0.csv"

echo -e "Making balanced dataset\n"
num_class1=$(wc -l $filename".class1.csv")
echo $num_class1

head -n $num_class1 $filename".class0.csv" > $filename".class0.matched.csv"

rm $filename".class0.csv"

cat $filename".class1.csv" < $filename".class0.matched.csv" | \
awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*$((2 * $threshold)), $0;}' | sort -n | cut -c8- \
> $filename".balanced.csv"

rm $filename".class0.matched.csv"

rm $filename".class1.csv"

# sbatch --export=ALL,INPUT=ChemInf_0.2 /cheminf/data/d2.full.csv ChemInf_0.2 /scripts/slurm/submit_cheminf_data_balancer_bash.sh
