#!/bin/bash

file=$1
filename=$(echo "${file%.*}")

echo -e "-------------------------------------------\n"
echo "START BALANCING OF: "$filename
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

# ./run_cheminf_data_balancing.sh file