#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=1
#SBATCH --error=/home/ha168/Semi_SuperviseAMC/Amc_Main/results/error.err
#SBATCH --job-name=semi6
#SBATCH --mem=60GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home/ha168/Semi_SuperviseAMC/Amc_Main/output.out
#SBATCH --partition=GPU
#SBATCH --signal=USR2@120
#SBATCH --time=4320
#SBATCH --nodelist=node038 # Specify 

# Navigate to the directory containing your Python script
cd /home/ha168/Semi_SuperviseAMC/Amc_Main
# Run your Python script
#python teacher_train_grid.py 
python main_train_grid.py
#python main_test.py 
#python validation_on_server.py
