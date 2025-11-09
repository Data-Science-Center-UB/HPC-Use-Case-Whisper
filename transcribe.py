# 1. Import packages & check CUDA
import os
from datetime import timedelta
import whisperx # Whisperx 
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# 2. Setup 
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    compute_type = "float16"  # Faster and more memory efficient on GPU
    batch_size = 16           # Adjust based on GPU memory
else:
    compute_type = "int8"     # Required or more efficient on CPU
    batch_size = 1            # Keep at 1 on CPU (larger values don’t help and may cause memory issues)

# 3. Select audio file
audio_file = "shortened_Buffy_Seas01-Epis01.en.wav"

# 4. Load Whisper model
language = "en"
model = whisperx.load_model("large-v3", 
                            device, 
                            compute_type=compute_type, 
                            language=language) # "tiny", "small", "large-v3", ...

# 5. Automatic Speech Recognition (ASR)
audio = whisperx.load_audio(audio_file)
asr_result = model.transcribe(audio, batch_size=batch_size)
#print(asr_result)

# 6. Word-level Forced Alignment
model_a, metadata = whisperx.load_align_model(language_code=asr_result["language"], device=device)
aligned_result = whisperx.align(asr_result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
#print(aligned_result)

# 7. Speaker Diarization
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=own_token, device=device)
diarization_result = diarize_model(audio,
                                   min_speakers=2, 
                                   max_speakers=2)
#print(diarization_result)

# 8. Merge outputs
final_result = whisperx.assign_word_speakers(diarization_result, aligned_result)
print(final_result) # segments are now assigned speaker IDs

# 9. Save Final Result
# Save as ...
method = "WhisperX"
file = "Buffy"
output_name = file + "_" + method + "_results"
txt_path = os.path.join(f"{output_name}.txt")

# Save 
with open(txt_path, "w", encoding="utf-8") as f:
    for seg in final_result["segments"]:
        speaker = seg.get("speaker")  # already present in your example
        text = seg["text"].strip()
        start = str(timedelta(seconds=seg["start"]))[:-3]
        f.write(f"{speaker}: {text} [{start}]\n\n")


print("Saved transcript to", txt_path)
