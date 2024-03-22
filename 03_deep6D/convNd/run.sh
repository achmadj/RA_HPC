#!/bin/bash
#SBATCH --account=student
#SBATCH --job-name=dl-6d
#SBATCH --partition=short
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64gb
#SBATCH --chdir=/mgpfs/home/ajaelani/_scratch/conv_6d/RA_HPC/03_deep6D/convNd
#SBATCH --output=/mgpfs/home/ajaelani/_scratch/conv_6d/RA_HPC/03_deep6D/convNd/slurm.out

python /mgpfs/home/ajaelani/_scratch/conv_6d/RA_HPC/03_deep6D/convNd/conv6d_h2.py