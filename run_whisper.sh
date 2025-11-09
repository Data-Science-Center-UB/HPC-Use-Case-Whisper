#!/bin/bash



# First, we need to tell SLURM some information about the job
#SBATCH --job-name=WHISPER

# Requesting 4 GB of memory - This is just a mall example,
# and this line allows the rest of the memory to be used
# by other jobs
#SBATCH --mem=4G

# Estimating a runtime of at most 10 minutes - Good estimates
# of runtime help the scheduler and may help your job get
# scheduled earlier, but will also kill your job if it exceeds
# the requested time
#SBATCH --time=00:10:00

# These 2 lines request a GPU
# If no GPU is needed, you can remove them
#SBATCH --partition=GPU

# Telling SLURM where to write output and error files
# %j gives job id
# %x gives job name
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err



# The actual code to be run starts here

# Load Conda
# Check available modules with `module avail`
module load anaconda/3.2021.11 
# Initialize Conda
source /hpc/opt/apps/Anaconda/3-2021.11/etc/profile.d/conda.sh 

# Activate the conda environment to get access to the required packages
conda activate whisperEnv

# Run the main script to generate predictions
Python transcribe.py

# Deactivate the virtual environment
conda deactivate
