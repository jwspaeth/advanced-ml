#!/bin/bash

#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --mem=2000
#SBATCH --output=job-output/subprocess-%j-stdout.txt
#SBATCH --error=job-output/subprocess--%j-stderr.txt
#SBATCH --time=7:00:00
#SBATCH --job-name=subprocess-%j
#SBATCH --mail-user=john.w.spaeth-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/jwspaeth/workspaces/baby_pattern_discovery/version-3/
#SBATCH --wait

python3 job_execution.py $@