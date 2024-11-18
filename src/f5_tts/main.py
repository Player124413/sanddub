import json
import subprocess
import os
import shutil
from pydub import AudioSegment
import random

# Load the JSON file
json_file_path = r"/content/s.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Safely fetch a value from a dictionary with a default
def safe_get(dialogue, key, default=None):
    value = dialogue.get(key)
    return value if value not in [None, ""] else default

# Construct CLI command from JSON
def build_command(dialogue, index, output_dir):
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

# Placeholder function to calculate similarity between two audio files
def calculate_similarity(audio1_path, audio2_path):
    # Replace with actual audio comparison logic
    return random.uniform(80, 100)

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

for index, dialogue in enumerate(data.get("dialogues", [])):
    temp_output_file = os.path.join(output_dir, "infer_cli_out.wav")
    final_output_file = os.path.join(output_dir, f"infer_cli_{index}.wav")
    ref_audio = safe_get(dialogue, "ref_audio", "")
    success = False
    attempt = 1

    while not success:
        print(f"Chunk {index}: Attempt {attempt} - Regenerating...")
        command = build_command(dialogue, index, output_dir)
        print(f"Executing: {command}")

        # Clean up the temporary output file
        remove_path(temp_output_file)
        remove_path(final_output_file)

        # Execute the command
        result = subprocess.run(command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        if result.returncode == 0 and os.path.exists(temp_output_file):
            # Rename the output file for easier identification
            os.rename(temp_output_file, final_output_file)
            print(f"Chunk {index}: Renamed output to {final_output_file}")

            similarity = calculate_similarity(final_output_file, ref_audio)
            print(f"Chunk {index}: Similarity = {similarity}%")

            if similarity >= 93:
                success = True
                output_files.append(final_output_file)
                print(f"Chunk {index} passed similarity check.")
            else:
                print(f"Chunk {index}: Similarity below threshold ({similarity}%). Retrying...")
                remove_path(final_output_file)
        else:
            print(f"Error in processing chunk {index}:")
            print(f"Return Code: {result.returncode}")
            print(f"Stdout: {result.stdout.decode()}")
            print(f"Stderr: {result.stderr.decode()}")
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
