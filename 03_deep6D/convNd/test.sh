#!/bin/bash
#SBATCH --account=student
#SBATCH --job-name=dl-6d
#SBATCH --partition=short
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64gb
#SBATCH --chdir=/mgpfs/home/ajaelani/_scratch/conv_6d/RA_HPC/03_deep6D/convNd
#SBATCH --output=/mgpfs/home/ajaelani/_scratch/conv_6d/RA_HPC/03_deep6D/convNd/test.out

python /mgpfs/home/ajaelani/_scratch/conv_6d/RA_HPC/03_deep6D/convNd/test.py