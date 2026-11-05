# HPC Use Case Whisper

The repository demonstrates how to use an HPC cluster (here, the DSC HPC cluster). It contains a use case to run a speech-to-text transcription and speaker diarization task with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [pyannote.audio](https://github.com/pyannote/pyannote-audio) on a GPU. See also [the GitHub repository Python-Whisper-Audio-Transcription](https://github.com/Data-Science-Center-UB/Python-Whisper-Audio-Transcription) for more information on this use case and its Jupyter notebooks. The example audio file is a tiny snippet from an audio file derived from [Lerner et al. 2022](https://huggingface.co/datasets/bazinga/bazinga/tree/main/data/BuffyTheVampireSlayer), licensed under CC-BY-NC-4.0. It contains speech from the series *Buffy the Vampire Slayer*, season 01, episode 01.

---

**Prerequisite:**

If you don't have a [Hugging Face account](https://huggingface.co/), you need to create one.

You will require a **token** later on for speaker diarization. You can create one by clicking your profile icon, then **Settings → Access Tokens**, and creating a token of type **Read**. Important: Don't share your token or commit it to GitHub.

In addition, pyannote requires you to **agree to share your contact information to access its models**. For that, go to the [pyannote speaker-diarization model](https://huggingface.co/pyannote/speaker-diarization-3.1) page, enter your information, and click **Agree and access repository**. Do the same for the [pyannote segmentation model](https://huggingface.co/pyannote/segmentation-3.0).

After setting up the environment in Step 2, log in on the cluster with `hf auth login`. In the notebook, use:

```python
from huggingface_hub import get_token
own_token = get_token()
```

This avoids saving the token in the notebook.

---

**Step 1 - Transfer the materials onto the cluster:**

After logging into the HPC cluster, go to a workspace directory, then clone the **Python-Whisper-Audio-Transcription** repository with:

```bash
git clone https://github.com/Data-Science-Center-UB/Python-Whisper-Audio-Transcription.git
cd Python-Whisper-Audio-Transcription
```

The notebooks are in `Notebooks/`, example audio files in `Data_Raw/`, and transcripts are saved in `Results/`. To use your own audio file, upload it to `Data_Raw/`.

**Step 2 - Create a new environment:**

Load Anaconda, create a Conda environment with Python 3.11, and install the packages from `requirements.txt`. The commands below reflect the tested DSC cluster setup. For workshops with many participants, installations should be carried out on a CPU compute node where required by cluster policy rather than placing heavy load on the login node.

```bash
module load anaconda/3.2021.11
source /hpc/opt/apps/Anaconda/3-2021.11/etc/profile.d/conda.sh

conda create -n whisper python=3.11 pip -y
conda activate whisper
python -m pip install -r requirements.txt
python -m ipykernel install --user --name whisper --display-name "whisper"
hf auth login
```

**Before running the job, check/adapt the notebook** `Notebooks/faster-whisper-pyannote.ipynb`:

1. In **Section 3.2**, select the input file and a name for the output, for example:

   ```python
   file_name = "File-A"
   audio_file = "../Data_Raw/File-A_buffy/shortened_Buffy_Seas01-Epis01.en.wav"
   ```

   Comment out other active file selections. The audio path is relative to the `Notebooks/` directory.

2. In **Section 1.2**, replace any token placeholder with the `get_token()` code given under Prerequisite.

3. For **GPU transcription**, confirm that Section 3.1 selects CUDA and add a check to avoid silently falling back to CPU:

   ```python
   device = "cuda" if torch.cuda.is_available() else "cpu"
   assert device == "cuda", "No CUDA GPU available"
   ```

   For faster processing, enable the existing batched inference line **after** creating the `WhisperModel` in Section 5.1, and pass the existing `batch_size` setting in Section 6:

   ```python
   model = BatchedInferencePipeline(model=model)
   ```

   ```python
   segments, info = model.transcribe(
       audio_16k,
       language="en",  # adapt or use None for automatic detection
       word_timestamps=False,
       vad_filter=True,
       beam_size=5,
       batch_size=batch_size
   )
   segments = list(segments)
   ```

4. For **GPU speaker diarization**, add the following immediately after `Pipeline.from_pretrained(...)` in Section 5.2:

   ```python
   diarization.to(torch.device(device))
   ```

   Adapt or omit `min_speakers` and `max_speakers` in Section 7 according to your audio. If your selected audio has more than two speakers, do not fix both parameters to `2`.

Save the edited notebook before submitting a job. The notebook's original save step writes to `Results/<file_name>.txt`; the `Results/` folder must exist.

**Step 3 - Run the Jupyter notebook by submitting a SLURM job:**

Submit the job so it runs on a GPU compute node. From the cloned repository root, create an `hpc/` directory (`mkdir -p hpc`), then create `hpc/run_notebook.sh` with the following content:

```bash
#!/bin/bash
#SBATCH --job-name=whisper
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --output=whisper-%j.out
#SBATCH --error=whisper-%j.err

set -e

module load anaconda/3.2021.11
source /hpc/opt/apps/Anaconda/3-2021.11/etc/profile.d/conda.sh
conda activate whisper

# Check that the job really has GPU access
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available()"

cd "$HOME/Python-Whisper-Audio-Transcription/Notebooks"

jupyter nbconvert --to notebook --execute "faster-whisper-pyannote.ipynb" \
    --ExecutePreprocessor.kernel_name=whisper \
    --ExecutePreprocessor.timeout=-1 \
    --output="faster-whisper-pyannote_executed.ipynb"
```

**Important:** `#SBATCH --gres=gpu:1` explicitly allocates a GPU. Using `--partition=GPU` alone did **not** make CUDA available in the tested setup.

From the repository root, submit the script:

```bash
sbatch hpc/run_notebook.sh
```

To check the job and its output (replace `JOBID` with the submitted job ID):

```bash
squeue -u "$USER"
cat whisper-JOBID.out
cat whisper-JOBID.err
```

The transcript is saved to `Results/<file_name>.txt`, and the executed notebook (including cell outputs) to `Notebooks/faster-whisper-pyannote_executed.ipynb`. If the job finishes but the expected transcript is missing, check that the edited notebook was saved and that the batch script executes the correct filename.

> **About the project**
>
> This repository was developed and delivered within the framework of the DataNord project at the Data Science Center of the University of Bremen. The DataNord project is funded under grant number 16DKZ2026A by the Federal Ministry of Research, Technology and Space (BMFTR) and financed by the European Union – NextGenerationEU. It is part of a broader initiative to strengthen data literacy among researchers by offering practical, hands-on training in core areas of 1) data science, 2) research data management (RDM), 3) critical thinking, and 4) ethical, legal and social aspects (ELSA). The materials were developed by Annika Nolte and Nele Fuchs.
