# HPC Use Case Whisper

The repository demonstrates how to use a HPC cluster. It contains a use case to run an speech-to-text transcription and speaker diarization task with [WhisperX](https://github.com/m-bain/whisperX). See also [the GitHub repository Python-Whisper-Audio-Transcription](https://github.com/Data-Science-Center-UB/Python-Whisper-Audio-Transcription) for more information on this use case. The audio file used in this use case is a tiny snippet from an audio file derived from [Lerner et al. 2022](https://huggingface.co/datasets/bazinga/bazinga/tree/main/data/BuffyTheVampireSlayer), licensed under CC-BY-NC-4.0. It contains speech from the series 'Buffy the Vampire Slayer", season 01, episode 01. 

---

**Prerequisite:**

If you don't have a [Hugging Face Account](https://huggingface.co/) you need to create one. 

You will require a **token** later on for the speaker diarization. You can create one by clicking on your profile icon; next click on "Access Tokens": Create a new access token of type "Read" or "Write". Important: Don't share your token.

In addition, pyannote requires you to **agree to share your contact information to access it's models**. For that, go on the [pyannote speaker-diarization model](https://huggingface.co/pyannote/speaker-diarization-3.1) page, enter your information, and click on "Agree and access repository". Do the same for the [pyannote segmentation model](https://huggingface.co/pyannote/segmentation-3.0).

You need to enter your token in the `transcribe.py` script by replacing `"ENTER_YOUR_TOKEN"` with your token in `own_token = "ENTER_YOUR_TOKEN"`.

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

# 2. Create the Conda environments
conda create -n whisperEnv python=3.10 -y

# 3. Run the installation via the sbatch script "setup.sh"
sbatch setup.sh
```

**Step 3 - Run the Python script by submitting a SLURM job**:
Submit the job so it runs on a compute node. Open `run_whisperx.sh` to see the exact steps.

```bash
sbatch run_whisperx.sh
```
