import os
import glob
from PIL import Image, ImageDraw, ImageFont

# Specify the directory that contains the heatmap images
directory = 'VIS'

# Get a list of all image files in the directory
image_files = glob.glob(os.path.join(directory, '*.png'))

def get_font_in_order(font_names, font_size):
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, font_size)
        except IOError:
            continue
    raise ValueError(f"None of the fonts {font_names} are available.")

def extract_text(filename):
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    parts = name_without_ext.split('_', 1)
    if len(parts) > 1:
        return parts[1]
    return name_without_ext

def draw_text_with_fallback(draw, text, position, font, fallback_fonts, fill):
    try:
        draw.text(position, text, font=font, fill=fill)
    except UnicodeEncodeError:
        fallback_font = get_font_in_order(fallback_fonts, font.size)
        draw.text(position, text, font=fallback_font, fill=fill)

# Unfortunately, the EMOJI doesn't work for some reason. 
# Check the filename for what CLIP "saw" if it saw an emoji.
primary_font_names = [
    "Segoe UI Emoji.ttf",  # Windows
    "Apple Color Emoji.ttc",  # macOS/iOS
    "NotoColorEmoji.ttf",  # Linux/Android
    "arialn.ttf",
    "DejaVuSansCondensed.ttf",
    "segoeui.ttf",
    "NotoSans-Regular.ttf",
    "symbola.ttf",
    "arial.ttf"
]
fallback_font_names = [
    "NotoColorEmoji.ttf",  # Common fallback for emojis
    "Apple Color Emoji.ttc",  # Another fallback for macOS/iOS
    "symbola.ttf"  # Fallback for symbols
]
font_size = 15

primary_font = get_font_in_order(primary_font_names, font_size)

for image_file in image_files:
    img = Image.open(image_file).convert('RGBA')
    draw = ImageDraw.Draw(img)

    # Extract the relevant part of the filename
    text_to_write = extract_text(image_file)

    # Write the extracted text into the image with fallback support for emojis
    draw_text_with_fallback(draw, text_to_write, (10, 10), primary_font, fallback_font_names, 'white')

    # Save the image, overwriting the original file
    img.save(image_file)

print('\n\n     3. Done writing filename overlay into images.')
