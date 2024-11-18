import json
import subprocess
import os
from pydub import AudioSegment  # You may need to install this library
import random

# Load the JSON file
json_file_path = r"/home/sangram/Desktop/aimodel/F5-TTS/src/f5_tts/infer/s.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Safely fetch a value from a dictionary with a default
def safe_get(dialogue, key, default=None):
    value = dialogue.get(key)
    return value if value not in [None, ""] else default

# Construct CLI command from JSON
def build_command(dialogue, index):
    base_command = ["f5-tts_infer-cli"]
    args = {
        "--model": "F5-TTS",
        "--ref_audio": safe_get(dialogue, "ref_audio", ""),
        "--ref_text": safe_get(dialogue, "ref_text", ""),
        "--gen_text": safe_get(dialogue, "text", ""),
        "--speed": safe_get(dialogue, "speed", 1.0),
        "--output_dir": f"output_{index}.wav",
    }

    # Add each argument and wrap values in double quotes
    for key, value in args.items():
        if value not in [None, ""]:  # Skip arguments with None or empty values
            base_command.append(f"{key} \"{value}\"")

    return " ".join(base_command)

# Placeholder function to calculate similarity between two audio files
def calculate_similarity(audio1_path, audio2_path):
    # For simplicity, this uses random values as similarity scores.
    # Replace this with actual logic (e.g., feature comparison).
    return random.uniform(80, 100)

# Process each dialogue entry
commands = []
output_files = []

for index, dialogue in enumerate(data.get("dialogues", [])):
    output_file = f"output_{index}.wav"
    ref_audio = safe_get(dialogue, "ref_audio", "")
    success = False
    retries = 3  # Max attempts to regenerate audio
    
    while not success and retries > 0:
        command = build_command(dialogue, index)
        print(f"Executing: {command}")
        result = subprocess.run(command, shell=True)
        
        if result.returncode == 0 and os.path.exists(output_file):
            similarity = calculate_similarity(output_file, ref_audio)
            print(f"Similarity for chunk {index}: {similarity}%")
            
            if similarity >= 93:
                success = True
                output_files.append(output_file)
            else:
                print(f"Chunk {index} similarity below threshold. Regenerating...")
                retries -= 1
        else:
            print(f"Error executing command for chunk {index}.")
            break

if all(os.path.exists(file) for file in output_files):
    # Combine all audio files
    with open("concat.txt", "w") as f:
        for file in output_files:
            f.write(f"file '{file}'\n")
    combine_command = "ffmpeg -f concat -safe 0 -i concat.txt -c copy output.wav"
    subprocess.run(combine_command, shell=True)
else:
    print("Some output files are missing or failed to meet the similarity threshold. Combination skipped.")
