# HPC Use Case Whisper

This repository demonstrates how to use an HPC cluster (here, the DSC HPC cluster). It describes how to run a speech-to-text transcription and speaker diarization task with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [pyannote.audio](https://github.com/pyannote/pyannote-audio) on a GPU. The notebooks, example audio files, and other materials are provided in the separate [Python-Whisper-Audio-Transcription](https://github.com/Data-Science-Center-UB/Python-Whisper-Audio-Transcription) repository; this repository contains the HPC instructions and a SLURM job script (`run_notebook.sh`).

---

**Prerequisite:**

For speaker diarization, you need a [Hugging Face account](https://huggingface.co/), a read-access token, and access to the required pyannote models. Detailed setup instructions are provided in the Jupyter notebooks in the [Python-Whisper-Audio-Transcription](https://github.com/Data-Science-Center-UB/Python-Whisper-Audio-Transcription) repository. Do not share or commit your token.

---

**Step 1 - Transfer the materials onto the cluster:**

After logging into the HPC cluster, go to a workspace directory, then clone both repositories (the HPC instructions and the transcription materials) alongside each other:

```bash
git clone https://github.com/Data-Science-Center-UB/HPC-Use-Case-Whisper.git
git clone https://github.com/Data-Science-Center-UB/Python-Whisper-Audio-Transcription.git
cd Python-Whisper-Audio-Transcription
```

The Jupyter notebooks are in `Notebooks/`, example audio files in `Data_Raw/`, and transcripts are saved in `Results/`. You can also upload your own audio file to `Data_Raw/`.

**Adapt the Jupyter notebook:**

Choose the Whisper notebook you want to run from the `Notebooks/` folder and check or adapt its settings including for GPU execution as described below.

>Note: Since Jupyter notebooks (`.ipynb`) are stored in JSON format, editing them directly in a terminal text editor on the cluster can be cumbersome. It may be easier to modify the notebook locally and then upload the updated file to the cluster, for example using `scp`. Make sure the SLURM script (`run_notebook.sh`) references the correct notebook filename.

1. **Select the audio file** in Section 3.2. Set `file_name` (used for the transcript filename) and `audio_file` (the path relative to `Notebooks/`), and comment out other active file selections. For example:

   ```python
   file_name = "File-A"
   audio_file = "../Data_Raw/File-A_buffy/shortened_Buffy_Seas01-Epis01.en.wav"
   ```

2. **Use your Hugging Face token:** In Section 1.2, replace the placeholder `own_token = # "ENTER_YOUR_TOKEN"`

3. **Use the GPU for transcription.** To use batched inference, ensure the following line is enabled **after** the `WhisperModel(...)` is created in Section 5.1:

   ```python
   model = BatchedInferencePipeline(model=model)
   ```

   In Section 6, pass `batch_size=batch_size` to `model.transcribe(...)` (the notebook sets `batch_size` in Section 3.1):

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

4. **Use the GPU for speaker diarization.** In Section 5.2, add the following directly after loading the pyannote pipeline with `Pipeline.from_pretrained(...)`:

   ```python
   diarization.to(torch.device(device))
   ```

In addition, there are further adaptations possible as explained inside the notebooks, such as adjusting the `min_speakers` and `max_speakers` settings in Section 7 for your recording.

**Step 2 - Create a new environment:**

Create a Conda environment (we name it "whisper") with Python 3.11 and install the packages listed in `requirements.txt`. The following module and paths refer to the DSC cluster setup.

```bash
module load anaconda/3.2021.11
source /hpc/opt/apps/Anaconda/3-2021.11/etc/profile.d/conda.sh
conda create -n whisper python=3.11 pip -y
conda activate whisper
python -m pip install -r requirements.txt
python -m ipykernel install --user --name whisper --display-name "whisper"
```

**Step 3 - Run the Jupyter notebook by submitting a SLURM job:**

Submit the job so it runs on a GPU compute node. The SLURM script [`run_notebook.sh`](run_notebook.sh) is already included in this HPC repository; open it to see the exact steps. It currently loads the `whisper` environment and executes `faster-whisper-pyannote.ipynb` from the other repository, using GPU.

From the **HPC-Use-Case-Whisper** directory, submit the job:

```bash
cd ../HPC-Use-Case-Whisper
sbatch run_notebook.sh
```

The script assumes that both repositories were cloned into the same workspace directory as shown in Step 1. If you use a different directory layout or notebook filename, adapt the path or filename in `run_notebook.sh` accordingly. The SLURM log files are written to the directory where you submitted the job.

To check whether the job is running, and to inspect its output (replace `JOBID` with the submitted job ID):

```bash
squeue -u "$USER"
cat whisper-JOBID.out
cat whisper-JOBID.err
```

The transcript is saved in `Results/<file_name>.txt`. The executed notebook (including cell outputs) is saved as `Notebooks/faster-whisper-pyannote_executed.ipynb`. If the transcript is missing, check that the source notebook was saved, the intended input file was selected, and the SLURM job completed successfully.

>About the project
>
>This repository was developed and delivered as part of workshops within the framework of the DataNord project at the Data Science Center of the University of Bremen. The DataNord project is funded under the grant number 16DKZ2026A by the Federal Ministry of Research, Technology and Space (BMFTR) and financed by the European Union – NextGenerationEU. It is part of a broader initiative to strengthen data literacy among researchers by offering practical, hands-on training in core areas of 1) data science, 2) research data management (RDM), 3) critical thinking, and 4) ethical, legal and social aspects (ELSA). The materials were developed by Annika Nolte (ORCID: 0000-0001-9562-0728) and Nele Fuchs (ORCID: 0009-0004-6837-6968).
