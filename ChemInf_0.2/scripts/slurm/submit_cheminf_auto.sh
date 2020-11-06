#!/bin/bash

while getopts i:o:c:n flag
do
    case "${flag}" in
        i) input_file=${OPTARG};;
        s) setup_file=${OPTARG};;
    esac
done

nr_jobs=$(($(wc -l config_overrider_file) - 1))
sbatch --array=0-$nr_jobs --export=ALL,INPUT=input_file,SETUP=setup_file \
sbatch_cheminf_auto.sh