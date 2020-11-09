#!/bin/bash

while getopts i: flag
do
    case "${flag}" in
        i) INFILE=${OPTARG};;
        *) echo "Invalid flag"
    esac
done

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


# ./scripts/label_data.sh -i cheminf/data/d2_trailset.csv