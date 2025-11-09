# HPC Use Case Whisper

This use case demonstrates how to use a HPC cluster to run an speech-to-text transcription and speaker diarization task with [WhisperX](https://github.com/m-bain/whisperX). See also [the GitHub repository Whisper-Audio-Transcription](https://github.com/Data-Science-Center-UB/Whisper-Audio-Transcription) for more information on this use case. The audio file used in this use case is a tiny snippet from an audio file derived from [Lerner et al. 2022](https://huggingface.co/datasets/bazinga/bazinga/tree/main/data/BuffyTheVampireSlayer), licensed under CC-BY-NC-4.0. It contains speech from the series 'Buffy the Vampire Slayer", season 01, episode 01. 

---

**Step 1 - Transfer the materials onto the cluster**:
After logging into the HPC cluster, go to a workspace directory, then clone the repository with:

```
git clone https://github.com/Data-Science-Center-UB/HPC-Use-Case-Whisper.git
```

**Step 2 - Create a new environment**:
Note that we run the environment installation on the CPU compute node to avoid putting too much load on the login node in hackathon/workshop settings where many people work in parallel. We therefore run the installation via the sbatch script `setup.sh` so it executes on a compute node, not on the login node. Open `setup.sh` to see the exact steps.

```
# 1. Enter the project folder
cd HPC-Use-Case-Whisper

# 2. Create the virtual environments
# The name is referenced in the bash scripts,
# so it should not be changed
python3 -m venv ./whisper-venv

# 3. Run the installation via the sbatch script "setup.sh"
sbatch setup.sh
```

**Step 3 - Run the Python script by submitting a SLURM job**:
Submit the job so it runs on a compute node. Open `run_whisperx.sh` to see the exact steps.

```bash
sbatch run_whisperx.sh
```
