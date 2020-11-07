#!/bin/bash

while getopts i: flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        *) echo "Invalid flag"
    esac
done

filename=$(echo "${infile%.*}")

echo -e "-------------------------------------------\n"
echo "START BALANCING OF: "$infile
echo -e "\n-------------------------------------------\n"

echo -e "Start counting data in file\n"
nlines=$(awk -F " " '{print $1}' $infile | wc -l);
threshold=$(echo "(0.01*$nlines) / 1" | bc)

echo -e "Extracting class 1\n"
head -n $threshold  $infile | awk -F"_" '$1=$1' OFS=" " | sort -k2 -n -t" " | sed "s/ /_/" | \
sed "s/\t/$(echo -e '\t')1$(echo -e '\t')/" | \
awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*$threshold, $0;}' | sort -n | cut -c8- \
> $filename"_class1.csv"

echo -e "Extracting class 0\n"
tail -n -$threshold  $infile | awk -F"_" '$1=$1' OFS=" " | sort -k2 -n -t" " | sed "s/ /_/" | \
sed "s/\t/$(echo -e '\t')0$(echo -e '\t')/" | \
awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*$(($nlines - $threshold)), $0;}' | sort -n | cut -c8- \
> $filename"_class0.csv"

cat $filename".class1.csv" < $filename".class0.csv" | \
awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*$((2 * $threshold)), $0;}' | sort -n | cut -c8- \
> $filename"_labeled.csv"

rm $filename"_class0.csv"

rm $filename"_class1.csv"

# ./scripts/slurm/label_data.sh -i cheminf/data/d2_full.csv