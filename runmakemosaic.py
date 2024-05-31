import os
import glob
import math
from PIL import Image, ImageDraw, ImageFont

# Specify the directory with the images to make a mosaic of
directory = 'VIS'

print('\n\n     4. Creating heatmap image mosaic(s)...\n')

# Get a list of all image files in the directory
image_files = glob.glob(os.path.join(directory, '*.png'))

# Group the images by the part of the filename before the first underscore
image_groups = {}
for image_file in image_files:
    base_name = os.path.basename(image_file)
    teh_name = base_name
    group_name = base_name.split('_')[0]
    if group_name not in image_groups:
        image_groups[group_name] = []
    image_groups[group_name].append(image_file)

# For each group, create a mosaic of images
for group_name, group_images in image_groups.items():
    images = [Image.open(img).convert('RGBA') for img in group_images]

    # Find the number of rows and columns for the square mosaic
    num_images = len(images)
    mosaic_size = math.ceil(math.sqrt(num_images))

    # Find the size of each individual image
    img_width, img_height = images[0].size

    # Create a new image for the mosaic
    mosaic = Image.new('RGBA', (img_width * mosaic_size, img_height * mosaic_size))

    # Paste each image into the mosaic
    for i in range(mosaic_size):
        for j in range(mosaic_size):
            # If we've run out of images, create a black tile
            if i * mosaic_size + j >= num_images:
                img = Image.new('RGBA', (img_width, img_height))
            else:
                img = images[i * mosaic_size + j]

            mosaic.paste(img, (j * img_width, i * img_height))

    # Save the mosaic
    mosaic.save(os.path.join(directory, f"{group_name}-MOSAIC.png"))
    print(f"Saved to {group_name}-MOSAIC.png")
