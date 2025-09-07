import os
import librosa
import torch
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from jiwer import wer
import numpy as np

# Paths to audio and transcript directories
audio_dir = "audio"
transcript_dir = "transcripts"

# Load audio files and corresponding transcripts
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
transcripts = []
for audio_file in audio_files:
    # Extract base name (e.g., 'TREMONT_audio' → 'TREMONT') and append '_transcript.txt'
    base_name = os.path.basename(audio_file).replace("_audio.wav", "")
    transcript_file = os.path.join(transcript_dir, f"{base_name}_transcript.txt")
    if os.path.exists(transcript_file):
        with open(transcript_file, "r") as f:
            transcripts.append(f.read().strip())
    else:
        raise FileNotFoundError(f"Transcript not found for {audio_file}: {transcript_file}")

# Ensure we have matching audio and transcripts
if len(audio_files) != len(transcripts):
    raise ValueError("Number of audio files and transcripts do not match!")

# Resample audio to 16kHz if needed
def resample_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

# Prepare dataset
def prepare_dataset(audio_files, transcripts):
    data = {"audio": [], "transcript": []}
    for audio_path, transcript in zip(audio_files, transcripts):
        audio = resample_audio(audio_path)
        data["audio"].append(audio)
        data["transcript"].append(transcript)
    return Dataset.from_dict(data)

# Tokenize and process audio with padding
def preprocess_data(batch, processor, max_length=200000):  # Max length in samples (~12.5s at 16kHz)
    audio = batch["audio"]
    # Pad or truncate audio to max_length
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.squeeze()
    # Ensure input_values is a tensor
    if not isinstance(input_values, torch.Tensor):
        input_values = torch.tensor(input_values, dtype=torch.float32)
    batch["input_values"] = input_values
    batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
    return batch

# Main fine-tuning function
def fine_tune_wav2vec2():
    # Load pre-trained model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
    
    # Custom data collator for padding
    def data_collator(features):
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]
        # Ensure input_values are tensors
        input_values = [v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32) for v in input_values]
        # Pad input_values
        input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)
        # Pad labels (token IDs)
        max_label_length = max(len(label) for label in labels)
        labels = [torch.tensor(label + [processor.tokenizer.pad_token_id] * (max_label_length - len(label)), dtype=torch.int64) for label in labels]
        labels = torch.stack(labels)
        return {"input_values": input_values, "labels": labels}
    
    # Prepare dataset
    dataset = prepare_dataset(audio_files, transcripts)
    dataset = dataset.map(lambda batch: preprocess_data(batch, processor), remove_columns=["audio", "transcript"])
    
    # Split into train and test (80-20 split)
    train_test = dataset.train_test_split(test_size=0.2)
    
    # Training arguments with default parameters
    training_args = TrainingArguments(
        output_dir="./wav2vec2-marine-asr",
        eval_strategy="steps",
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=30,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id="aishwaryamy/wav2vec2-marine-asr",
        report_to="none"
    )
    
    # Initialize trainer with custom data collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test["train"],
        eval_dataset=train_test["test"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor)
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"Test WER: {eval_results['eval_wer']:.2f}")
    
    # Inference on a sample
    sample_audio = resample_audio(audio_files[0])
    inputs = processor(sample_audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print(f"Sample transcription: {transcription}")
    
    # Push to Hugging Face Hub
    trainer.push_to_hub()

# Compute WER metric
def compute_metrics(pred, processor):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
    return {"wer": wer(label_str, pred_str)}

if __name__ == "__main__":
    fine_tune_wav2vec2()