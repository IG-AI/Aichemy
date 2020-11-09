#!/bin/bash

#SBATCH --job-name=caveman
#SBATCH --time=3:00:00

#SBATCH --nodes=1
#SBATCH --mem=100G

python caveman4thewin.py $1 $2 $3
