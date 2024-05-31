import argparse
import subprocess
import os
import sys
import threading
import time

# 1: Check if the folders "VIS", "TOK", and "IMG_IN" exist. If not, create them in the current directory
directories = ["VIS", "TOK", "IMG_IN"]
for dir_name in directories:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# 2: Check if the "IMG_IN" folder is empty
while not os.listdir("IMG_IN"):
    input("Please put some images into the IMG_IN folder. Press enter to continue...")
    
print('\n     1. Running Long-CLIP gradient ascent; this may take 30s to 2min / image, depending on model + hardware...\n')
parser = argparse.ArgumentParser(description='Process all images in a directory using Long-CLIP gradient ascent.')
parser.add_argument('--image_dir', type=str, default="IMG_IN", help='The directory containing the images.')
args = parser.parse_args()
if args.image_dir is None:
    raise ValueError("You must provide a path to the image folder using the argument: --image_dir \"path/to/image/folder\"")

# Get a list of all files in the directory "IMG_IN"
image_files = os.listdir(args.image_dir)

# Loop over each file
for image_file in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(args.image_dir, image_file)
   
    try:
        # Call the gradient ascent CLIP script with the image path as an argument
        clip_command = ["python", f"longclipga_AMP-finetune.py", "--image_path", image_path]
        result = subprocess.run(clip_command, check=True, stdout=None, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    except KeyboardInterrupt:
        print("\nProcess interrupted by the user.")
        sys.exit(1)  

    if result.returncode == 0:
        output_filename = f"TOK/tokens_{os.path.splitext(os.path.basename(image_path))[0]}.txt"
        print(f"CLIP tokens saved to {output_filename}.")
        # Continue with processing the output tokens
    else:
        print("\nCLIP script encountered an error (below). Continuing with next image (if applicable)...\n\n")
        print("Error details:", result.stderr)
        
        
print("Running Long-CLIP attention visualization... This should only take a minute or two for all images combined.")

# Execute the rest of the scripts in order
scripts_to_run = ["longclip-explain-attention-visualization.py", "runmakenames.py", "runmakemosaic.py"]
for script in scripts_to_run:
    process = subprocess.run(["python", script], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    
# 4: Print the final message
print("\n\nDONE. Check the 'VIS' folder for the results!")