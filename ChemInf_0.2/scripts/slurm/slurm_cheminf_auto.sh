#!/bin/bash

while getopts i:o:c:n flag
do
    case "${flag}" in
        i) input_file=${OPTARG};;
        c) config_overrider_file=${OPTARG};;
        n) names_file=${OPTARG};;
    esac
done

nr_jobs=$(($(wc -l config_overrider_file) - 1))
sbatch --array=0-$nr_jobs --export=ALL,INPUT=input_file,CONFIGS=config_overrider_file,NAMES=names_file \
sbatch_cheminf_auto.sh