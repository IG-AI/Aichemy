#!/bin/bash

work_dir=/proj/carlssonlab/users/x_danag/ChemInf/ChemInf_0.2

cd $work_dir

filename=$(echo "${INFILE%.*}")

echo -e "-------------------------------------------\n"
echo "START LABELING OF: "$INFILE
echo -e "\n-------------------------------------------\n"

#echo -e "Start sorting data in file\n"
#awk -F"_" '$1=$1' OFS="\t" $INFILE | sort -k2 -n -t$'\t' | sed "s/\t/_/" > $filename"_sorted.csv"

echo -e "Start counting data in file\n"
nlines=$(awk -F "\t" '{print $1}' $INFILE | wc -l);
threshold=$(echo "(0.01*$nlines) / 1" | bc)

#echo -e "Extracting class 1\n"
#head -n $threshold  $filename"_sorted.csv" | sed "s/\t/$(echo -e '\t')1$(echo -e '\t')/" | \
#awk 'BEGIN{srand();} {printf "%06d %s\n", $filenamerand()*$threshold, $0;}' | sort -n | cut -c8- \
#> $filename"_class1.csv"

nr_class0=$(($nlines - $threshold))

echo -e "Extracting class 0\n"
tail -n -$threshold  $filename"_sorted.csv" | sed "s/\t/$(echo -e '\t')0$(echo -e '\t')/" | \
awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*$nr_class0, $0;}' | sort -n | cut -c8- \
> $filename"_class0.csv"

cat $filename"_class1.csv" < $filename"_class0.csv" | \
awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*$nlines, $0;}' | sort -n | cut -c8- \
> $filename"_labeled.csv"

# ./scripts/label_data.sh -i cheminf/data/d2_full.csv