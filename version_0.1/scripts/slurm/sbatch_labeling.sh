full_file=$1
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
cut -f1 $full_file | awk -F"_" '$1=$1' OFS="\t" | sort -t$'\t' -k2,2n -V | \
sed "s/\t/_/" | head -n$threshold > $filename"_"$percentage"_class1.csv"

echo -e "Start labeling data\n"
python scripts/labeling.py $filename"_"$percentage"_class1.csv" $full_file $filename"_"$percentage".csv"
