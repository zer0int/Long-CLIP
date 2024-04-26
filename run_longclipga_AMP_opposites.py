import argparse
import subprocess
import os
import sys
import threading
import time
from colorama import Fore, Style

image_dir = "IMG_IN"

# Get a list of all files in the directory
image_files = os.listdir(image_dir)

print(Fore.RED + "\nRunning CLIP gradient ascent (cosine similarity)..." + Style.RESET_ALL)
# Loop over each file
for image_file in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(image_dir, image_file)
    print(Fore.GREEN + f"Now computing: {image_file}" + Style.RESET_ALL)
    try:
        # Call the gradient ascent CLIP script with the image path as an argument
        clip_command = ["python", f"longclipga_AMP.py", "--image_path", image_path]
        result = subprocess.run(clip_command, stdout=None, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    except KeyboardInterrupt:
        stop_thread = True
        t.join()
        print("\nProcess interrupted by the user.")
        sys.exit(1)  

    if result.returncode == 0:
        print(f"CLIP opinion tokens saved to the 'TOK' folder.")
        # Continue with processing the output tokens
    else:
        print("\nCLIP script encountered an error (below). Continuing with next image (if applicable)...\n\n")
        print("Error details:", result.stderr)

print(Fore.RED + "\nRunning CLIP gradient ascent (cosine DIS-similarity)..." + Style.RESET_ALL)
        
for image_file in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(image_dir, image_file)
    print(Fore.GREEN + f"Now computing: {image_file}" + Style.RESET_ALL)
    try:
        # Call the gradient ascent CLIP script with the image path as an argument
        clip_command = ["python", f"longclipga_AMP_anti.py", "--image_path", image_path]
        result = subprocess.run(clip_command, stdout=None, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    except KeyboardInterrupt:
        stop_thread = True
        t.join()
        print("\nProcess interrupted by the user.")
        sys.exit(1)  

    if result.returncode == 0:
        print(f"CLIP opinion tokens saved to the 'TOK' folder.")
        # Continue with processing the output tokens
    else:
        print("\nCLIP script encountered an error (below). Continuing with next image (if applicable)...\n\n")
        print("Error details:", result.stderr)
