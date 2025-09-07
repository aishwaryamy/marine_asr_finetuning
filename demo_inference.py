import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

# Load fine-tuned model and processor
model = Wav2Vec2ForCTC.from_pretrained("aishwaryamy/wav2vec2-marine-asr")
processor = Wav2Vec2Processor.from_pretrained("aishwaryamy/wav2vec2-marine-asr")
speech, rate = librosa.load("audio/WHITE_WATER_audio.wav", sr=16000)
input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
with torch.no_grad():
    logits = model(input_values).logits
pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(pred_ids)[0]
print(f"Transcription: {transcription}")