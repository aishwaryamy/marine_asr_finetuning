Marine ASR Fine-Tuning Project
Overview
This project fine-tunes a Wav2Vec 2.0 model (facebook/wav2vec2-base) for Automatic Speech Recognition (ASR) on a small dataset of marine-related audio files. The goal is to transcribe audio recordings (e.g., marine radio communications) with high accuracy. The project uses the Hugging Face Transformers library and processes .wav audio files with corresponding transcript files.
Progress

Dataset:

Audio Files: The audio/ directory contains:
WHITE_WATER_audio.wav
SUSIE_ROSE_audio.wav
TREMONT_audio.wav
AFTER_HOURS_audio.mp3 (requires conversion to .wav for use)


Transcripts: The transcripts/ directory contains:
WHITE_WATER_transcript.txt
SUSIE_ROSE_transcript.txt
TREMONT_transcript.txt
Possibly BARGE_U-1512_transcripts (needs renaming to BARGE_U-1512_transcript.txt if used with a corresponding .wav file)


Note: A BARGE_U-1512_audio/ directory was referenced but does not exist. If a BARGE_U-1512_audio.wav file is available, ensure it’s in audio/ and paired with BARGE_U-1512_transcript.txt.


Scripts:

fine_tune_wav2vec2_marine.py:
Fine-tunes the Wav2Vec 2.0 model using the Hugging Face Trainer API.
Loads .wav files from audio/ and matching *_transcript.txt files from transcripts/.
Resamples audio to 16kHz, pads/truncates to a fixed length (200,000 samples ~12.5s), and processes transcripts.
Uses default parameters: batch size 8, 30 epochs, 3e-4 learning rate.
Splits data into 80% training, 20% testing.
Computes Word Error Rate (WER) and uploads the model to Hugging Face Hub (yourusername/wav2vec2-marine-asr).
Status: Fixed issues with tensor conversion, processor scope, and MPS backend (ctc_loss not supported on Mac M1/M2). Ready to run with PYTORCH_ENABLE_MPS_FALLBACK=1 locally or on Google Colab with GPU.


demo_inference.py:
Tests the fine-tuned model on audio/WHITE_WATER_audio.wav.
Loads the model from yourusername/wav2vec2-marine-asr and outputs a transcription.
Status: Ready to use post-training.




Issues Encountered and Fixed:

NameError in data_collator: Fixed by moving data_collator inside fine_tune_wav2vec2 to access processor.
MPS Backend Error: aten::_ctc_loss not implemented on Mac M1/M2. Resolved by setting PYTORCH_ENABLE_MPS_FALLBACK=1 for CPU fallback.
BARGE_U-1512_audio: Directory audio/BARGE_U-1512_audio/ does not exist. Skipped for now; confirm if BARGE_U-1512_audio.wav is available.
Git LFS Setup: .gitattributes was missing; instructions provided to create it for large audio files.


Current Status:

The project is ready to be fine-tuned on at least 3 audio files (WHITE_WATER_audio.wav, SUSIE_ROSE_audio.wav, TREMONT_audio.wav) with their transcripts.
Expected WER: ~20-40% due to the small dataset.
Local training on Mac (M1/M2) takes ~30-50 minutes for 3 files with CPU fallback.
Google Colab with T4 GPU is recommended for faster training (~10-15 minutes for 3 files).
Repository setup is in progress, with Git LFS configured for large audio files.



Repository Structure
marine_asr_finetuning/
├── audio/
│   ├── WHITE_WATER_audio.wav
│   ├── SUSIE_ROSE_audio.wav
│   ├── TREMONT_audio.wav
│   ├── AFTER_HOURS_audio.mp3
├── transcripts/
│   ├── WHITE_WATER_transcript.txt
│   ├── SUSIE_ROSE_transcript.txt
│   ├── TREMONT_transcript.txt
│   ├── BARGE_U-1512_transcripts (if exists, rename to BARGE_U-1512_transcript.txt)
├── fine_tune_wav2vec2_marine.py
├── demo_inference.py
├── .gitattributes
├── .gitignore

Setup Instructions
Prerequisites

Local Machine:

Python 3.12
Install dependencies:pip install "transformers==4.44.2" "datasets>=2.14.0" librosa soundfile jiwer torch huggingface_hub


Install ffmpeg for audio conversion:brew install ffmpeg


Install Git LFS:brew install git-lfs
git lfs install


Hugging Face account and token: https://huggingface.co/settings/tokens


Google Colab (recommended for faster training):

Access: https://colab.research.google.com/
Enable T4 GPU: Runtime → Change runtime type → GPU → T4 GPU → Save



Local Setup

Clone the Repository:
git clone https://github.com/yourusername/marine_asr_finetuning.git
cd marine_asr_finetuning
git lfs pull


Activate Virtual Environment (if used):
python3 -m venv env
source env/bin/activate


Install Dependencies:
pip install "transformers==4.44.2" "datasets>=2.14.0" librosa soundfile jiwer torch huggingface_hub


Convert MP3 Files (if including AFTER_HOURS_audio.mp3):
ffmpeg -i audio/AFTER_HOURS_audio.mp3 -ac 1 -ar 16000 audio/AFTER_HOURS_audio.wav


Handle BARGE_U-1512 (if applicable):

If BARGE_U-1512_audio.wav exists in audio/:ffmpeg -i audio/BARGE_U-1512_audio.wav -ac 1 -ar 16000 audio/BARGE_U-1512_audio.wav
mv transcripts/BARGE_U-1512_transcripts transcripts/BARGE_U-1512_transcript.txt


If not, skip it.


Update Hugging Face Model ID:

Edit fine_tune_wav2vec2_marine.py and demo_inference.py to replace yourusername with your Hugging Face username:sed -i '' 's/yourusername/your_actual_username/' fine_tune_wav2vec2_marine.py
sed -i '' 's/yourusername/your_actual_username/' demo_inference.py




Run Fine-Tuning:
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 fine_tune_wav2vec2_marine.py


Training Time: ~30-50 minutes for 3 files; ~5-10 minutes per additional file.
Output: WER (~20-40%), sample transcription, model uploaded to https://huggingface.co/yourusername/wav2vec2-marine-asr.


Test the Model:
python3 demo_inference.py



Google Colab Setup (Recommended)

Open Colab:

Go to https://colab.research.google.com/ and create a new notebook.


Enable GPU:

Click Runtime → Change runtime type → Select GPU → Choose T4 GPU → Save.
Verify:!nvidia-smi




Clone the Repository:
!git clone https://github.com/yourusername/marine_asr_finetuning.git
%cd marine_asr_finetuning
!git lfs pull


Install Dependencies:
!pip install "transformers==4.44.2" "datasets>=2.14.0" librosa soundfile jiwer torch huggingface_hub
!huggingface-cli login


Paste your Hugging Face token.


Run Fine-Tuning:
!python3 fine_tune_wav2vec2_marine.py


Training Time: ~10-15 minutes for 3 files; ~5 minutes per additional file.


Test the Model:
!python3 demo_inference.py



Next Steps

Add More Data:
Convert additional audio files (e.g., AFTER_HOURS_audio.mp3) to .wav and ensure matching transcripts.
Check for BARGE_U-1512_audio.wav or other missing files.


Optimize Training:
Reduce max_length in fine_tune_wav2vec2_marine.py (e.g., to 100000) or set per_device_train_batch_size=4 for memory issues.
Experiment with hyperparameters (e.g., learning rate, epochs) for better WER.


Verify BARGE_U-1512:
Run:ls -l audio/BARGE_U*
ls -l transcripts/BARGE_U-1512*


Share output to confirm if BARGE_U-1512_audio.wav exists or needs recovery.



Troubleshooting

BARGE_U-1512: If audio/BARGE_U-1512_audio.wav is missing, skip it or locate the file. Ensure transcripts/BARGE_U-1512_transcript.txt matches.
Memory Issues: In Colab, reduce max_length or batch size in fine_tune_wav2vec2_marine.py.
Git Issues: Ensure large files are tracked with Git LFS:git lfs ls-files
git lfs push --all origin main



Contact the project owner for the Hugging Face username and GitHub repository details (https://github.com/yourusername/marine_asr_finetuning).
