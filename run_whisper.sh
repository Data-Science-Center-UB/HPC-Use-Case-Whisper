#!/bin/bash

# SLURM job settings
#SBATCH --job-name=whisper
#SBATCH --partition=GPU           # Allocate GPU
#SBATCH --gres=gpu:1              
#SBATCH --time=02:00:00           # Maximum runtime: 2 hours
#SBATCH --mem=24G                 # Request 24 GB of RAM
#SBATCH --output=whisper-%j.out   # Standard output (%j = job ID)

# Load Anaconda and activate the environment with required packages
module load anaconda/3.2021.11
source /hpc/opt/apps/Anaconda/3-2021.11/etc/profile.d/conda.sh
conda activate whisper

# Navigate to the notebook directory
cd "$SLURM_SUBMIT_DIR/../Python-Whisper-Audio-Transcription/Notebooks" || exit 1

# Execute the notebook using the 'whisper' kernel
# Save the executed notebook, including its outputs, as a separate file
jupyter nbconvert --to notebook --execute "faster-whisper-pyannote.ipynb" \
  --ExecutePreprocessor.kernel_name=whisper \
  --ExecutePreprocessor.timeout=-1 \
  --output="faster-whisper-pyannote_executed.ipynb"
