import json
import subprocess
import os

# Load the JSON file
json_file_path = "/content/s.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Template for the command
command_template = (
    "python -m f5_tts_mlx.generate "
    "--text \"{text}\" "
    "--duration {duration} "
    "--ref-audio \"{ref_audio}\" "
    "--ref-text \"{ref_text}\" "
    "--steps 32 "
    "--method {method} "
    "--cfg {cfg} "
    "--sway-coef {sway_coef} "
    "--speed {speed} "
    "--seed {seed} "
    "--output {output_file}"
)

# Function to safely fetch a value with a default
def safe_get(dialogue, key, default):
    return dialogue.get(key, default) if dialogue.get(key) not in [None, ""] else default

# Process each dialogue entry
commands = []
output_files = []

for index, dialogue in enumerate(data.get("dialogues", [])):
    output_file = f"output_{index}.wav"
    command = command_template.format(
        text=safe_get(dialogue, "text", "Placeholder text"),
        duration=safe_get(dialogue, "duration", 1.0),
        ref_audio=safe_get(dialogue, "ref_audio", "default_audio.wav"),
        ref_text=safe_get(dialogue, "ref_text", "Placeholder reference text"),
        method=safe_get(dialogue, "method", "default"),
        cfg=safe_get(dialogue, "cfg", 1.0),
        sway_coef=safe_get(dialogue, "sway-coef", 1.0),
        speed=safe_get(dialogue, "speed", 1.0),
        seed=safe_get(dialogue, "seed", -1),
        output_file=output_file
    )
    commands.append((command, output_file))
    output_files.append(output_file)

# Execute the commands
for command, output_file in commands:
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error executing command: {command}")
    elif not os.path.exists(output_file):
        print(f"Output file {output_file} was not created.")

# Combine audio files
if all(os.path.exists(file) for file in output_files):
    with open("concat.txt", "w") as f:
        for file in output_files:
            f.write(f"file '{file}'\n")
    combine_command = "ffmpeg -f concat -safe 0 -i concat.txt -c copy output.wav"
    subprocess.run(combine_command, shell=True)
else:
    print("Some output files are missing. Audio combination skipped.")
