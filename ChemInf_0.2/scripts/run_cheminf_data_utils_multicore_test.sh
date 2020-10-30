#!/bin/bash

SRC=$HOME"/PycharmProjects/ChemInf"
INFILE="cheminf_0.1_da/cheminf/data/d2.trialset.csv"
OUTFILE="cheminf_0.1_da/cheminf/data/d2.trialset.test.csv"
LOG_DIR="cheminf_0.1_da/logs"

conda activate ChemInf
cd $SRC
rm $OUTFILE

for i in 12 10 8 6 4 2 1
do
  echo -e "Starting test with "$i" cores\n"

  python cheminf_0.1_da/scripts/cheminf_data_utils.py -i $INFILE -o $OUTFILE -m resample -nc $i \
  > $LOG_DIR"/test_data_utils_nc"$i".log"

  rm $OUTFILE
done

# bash -i cheminf_0.1_da/scripts/run_cheminf_data_utils_multicore_test.sh