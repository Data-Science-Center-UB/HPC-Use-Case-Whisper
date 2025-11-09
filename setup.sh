#!/bin/bash



# First, we need to tell SLURM some information about the job
#SBATCH --job-name=whisper-setup

# Requesting 10 GB of memory - This should be enough for setup
# and allows 12 concurrent users on a 128 GB node
#SBATCH --mem=10G

# Estimating a runtime of at most 15 minutes - Good estimates
# of runtime help the scheduler and may help your job get
# scheduled earlier, but will also kill your job if it exceeds
# the requested time
#SBATCH --time=00:15:00

# Running on CPU only - setup does not need a GPU
# We use 3 CPU cores
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=3

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

# Activate the environment
conda activate whisperEnv

# Install required Python packages
conda install -c conda-forge ffmpeg pip -y
pip install whisperx

# Deactivate the virtual environment
conda deactivate
