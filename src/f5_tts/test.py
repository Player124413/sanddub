import json
import subprocess
import os
import shutil
from pydub import AudioSegment
import whisper
from scipy.spatial.distance import cosine
import torch
from transformers import pipeline
import numpy as np  # Add this for numpy usage


# Hugging Face Token (Replace with your actual token)
HUGGINGFACE_TOKEN = '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'  # Replace with your token if needed

# Load the JSON file
json_file_path = r"/content/s.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU

# Load Hugging Face Speech Emotion Recognition model
emotion_model = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    device=device,
    use_auth_token=HUGGINGFACE_TOKEN
)

# Safely fetch a value from a dictionary with a default
def safe_get(dialogue, key, default=None):
    return dialogue.get(key, default)

# Construct CLI command from JSON
def build_command(dialogue, output_dir):
    base_command = ["f5-tts_infer-cli"]
    args = {
        "--model": "F5-TTS",
        "--ref_audio": safe_get(dialogue, "ref_audio", ""),
        "--ref_text": safe_get(dialogue, "ref_text", ""),
        "--gen_text": safe_get(dialogue, "text", ""),
        "--speed": safe_get(dialogue, "speed", 1.0),
        "--output_dir": output_dir,
    }

    for key, value in args.items():
        if value not in [None, ""]:
            base_command.append(f'{key} "{value}"')

    return " ".join(base_command)

# Preprocess audio for Whisper
def preprocess_audio(model, audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    return whisper.log_mel_spectrogram(audio).to(model.device)

# Calculate similarity using Whisper embeddings
def calculate_whisper_similarity(model, audio_path1, audio_path2):
    mel1 = preprocess_audio(model, audio_path1)
    mel2 = preprocess_audio(model, audio_path2)

    with torch.no_grad():
        embedding1 = model.encoder(mel1.unsqueeze(0)).squeeze(0).mean(dim=0).cpu().numpy()
        embedding2 = model.encoder(mel2.unsqueeze(0)).squeeze(0).mean(dim=0).cpu().numpy()

    return 1 - cosine(embedding1, embedding2)

# Emotion prediction using Hugging Face pipeline
def predict_emotion(audio_path):
    """
    Predict emotion from audio file using Hugging Face pipeline.
    """
    results = emotion_model(audio_path)
    # Take the emotion with the highest score
    emotion = max(results, key=lambda x: x['score'])['label']
    return emotion

# Extract audio features (e.g., pitch, tempo, MFCC)
def extract_audio_features(audio_path):
    """
    Extract pitch, tempo, and MFCC features from an audio file.
    Replace with actual implementation using librosa or another library.
    """
    # Dummy data for demonstration purposes
    pitch = np.random.rand(5000)
    tempo = np.random.uniform(60, 180)  # Random tempo in BPM
    mfccs = np.random.rand(5000)
    return pitch, tempo, mfccs

# Main similarity calculation function
def calculate_similarity(audio1_path, audio2_path):
    """
    Calculate similarity between two audio files based on their emotional content
    and extracted features (pitch, tempo, MFCC).
    """
    # Step 1: Emotion prediction
    emotion1 = predict_emotion(audio1_path)
    emotion2 = predict_emotion(audio2_path)
    print(f"Emotion 1: {emotion1}, Emotion 2: {emotion2}")

    # Step 2: Extract audio features (pitch, tempo, MFCCs)
    pitch1, tempo1, mfccs1 = extract_audio_features(audio1_path)
    pitch2, tempo2, mfccs2 = extract_audio_features(audio2_path)

    print(f"Pitch1 length: {len(pitch1)}, Pitch2 length: {len(pitch2)}")
    print(f"MFCCs1 length: {len(mfccs1)}, MFCCs2 length: {len(mfccs2)}")
    print(f"Tempo1: {tempo1}, Tempo2: {tempo2}")

    # Step 3: Standardize lengths (if needed)
    def standardize_length(array1, array2, target_length=5000):
        # Truncate to minimum length or target length
        min_len = min(len(array1), len(array2), target_length)
        print(f"Standardizing to length: {min_len}")
        array1 = array1[:min_len]
        array2 = array2[:min_len]
        return array1, array2

    # Standardize pitch and MFCC lengths
    pitch1, pitch2 = standardize_length(pitch1, pitch2)
    mfccs1, mfccs2 = standardize_length(mfccs1, mfccs2)

    # Step 4: Compute cosine similarities for pitch and MFCC
    print(f"Computing cosine similarity for pitch...")
    pitch_similarity = 1 - cosine(pitch1, pitch2) if len(pitch1) > 0 else 0
    print(f"Pitch Similarity: {pitch_similarity}")

    print(f"Computing cosine similarity for MFCCs...")
    mfcc_similarity = 1 - cosine(mfccs1, mfccs2) if len(mfccs1) > 0 else 0
    print(f"MFCC Similarity: {mfcc_similarity}")

    # Tempo similarity is a scalar comparison
    tempo_similarity = 1 if abs(tempo1 - tempo2) < 5 else 0
    print(f"Tempo Similarity: {tempo_similarity}")

    # Combine feature similarities
    audio_similarity = (pitch_similarity + mfcc_similarity + tempo_similarity) / 3
    print(f"Audio Similarity: {audio_similarity}")

    # Step 5: Combine emotional similarities (1 if emotions match, otherwise 0)
    emotion_similarity = 1 if emotion1 == emotion2 else 0
    print(f"Emotion Similarity: {emotion_similarity}")

    # Final similarity score
    final_similarity = (audio_similarity + emotion_similarity) / 2
    print(f"Final Similarity: {final_similarity}")

    return final_similarity, emotion1, emotion2, audio_similarity, emotion_similarity

# Remove file or directory
def remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

# Process each dialogue entry
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

output_files = []
model = whisper.load_model("base")  # Load Whisper model once

for index, dialogue in enumerate(data.get("dialogues", [])):
    temp_output_file = os.path.join(output_dir, "infer_cli_out.wav")
    final_output_file = os.path.join(output_dir, f"infer_cli_{index}.wav")
    ref_audio = safe_get(dialogue, "ref_audio", "")

    success = False
    attempt = 1

    while not success:
        print(f"Chunk {index}: Attempt {attempt} - Regenerating...")
        command = build_command(dialogue, output_dir)
        print(f"Executing: {command}")

        # Clean up the temporary output file
        remove_path(temp_output_file)
        remove_path(final_output_file)

        result = subprocess.run(command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        if result.returncode == 0 and os.path.exists(temp_output_file):
            os.rename(temp_output_file, final_output_file)
            print(f"Chunk {index}: Renamed output to {final_output_file}")

            # Calculate similarity using Whisper embeddings
            whisper_similarity = calculate_whisper_similarity(model, final_output_file, ref_audio)
            print(f"Chunk {index}: Whisper Similarity = {whisper_similarity:.2f}")

            # Calculate the actual similarity using emotional and audio features
            final_similarity, emotion1, emotion2, audio_similarity, emotion_similarity = calculate_similarity(final_output_file, ref_audio)
            print(f"Chunk {index}: Audio Similarity = {audio_similarity:.2f}, Emotion Similarity = {emotion_similarity:.2f}, Final Similarity = {final_similarity:.2f}")

            # Check if both similarities meet the threshold and emotions match
            if final_similarity >= 0.93 and whisper_similarity >= 0.93:
                success = True
                output_files.append(final_output_file)
                print(f"Chunk {index} passed similarity and emotion checks.")
            else:
                print(f"Chunk {index}: One or both similarities below threshold, retrying...")
                remove_path(final_output_file)
        else:
            print(f"Error processing chunk {index}: {result.stderr.decode()}")
            break

        attempt += 1

# Combine all successful output files
if output_files:
    combined_audio = AudioSegment.empty()
    for file in output_files:
        combined_audio += AudioSegment.from_file(file)

    combined_audio.export("output.wav", format="wav")
    print("Combined audio saved as output.wav")
else:
    print("No audio chunks generated successfully. Combination skipped.")
