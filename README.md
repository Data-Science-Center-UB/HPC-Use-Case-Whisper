# HPC-Use-Case-Whisper

[Whisper ... here: Whisperx]

**Step 1:** After logging into the HPC cluster, activate Conda and create a new environment for Whisperx. Afterwards we activate this new environment and install necessary packages. 

::: info 
Because our cluster’s NVIDIA driver currently only supports CUDA 11.x, to run WhisperX on the GPU you would need to install CUDA-11–compatible CTranslate2 (e.g., 3.24.0) and faster-whisper (e.g., 0.10.1), PyTorch 1.10.2 with cudatoolkit=11.1, then install WhisperX 3.1.1 with --no-deps so it doesn’t upgrade PyTorch. 
:::

   ```
   conda create -n whisperx-env python=3.10 -y
   conda activate whisperx-env
   conda install -c conda-forge ffmpeg pip -y
   pip install whisperx
   ```
   
**Step 2:** The code to import an example dataset (file "Buffy.wav") and perform the analysis inside a Python script (file "test_Whisperx.py") is:

```
```
