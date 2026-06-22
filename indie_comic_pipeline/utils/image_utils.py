"""
IMAGE UTILITIES
Helper functions for processing, resizing, and validating images
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = getattr(Image, 'LANCZOS')

def resize_image(image_path, output_path, size=(512, 512)):

    """Resize image to specified dimensions"""

    img = Image.open(image_path)

    img_resized = img.resize(size, LANCZOS)

    img_resized.save(output_path)

    return output_path

def create_comic_grid(images_paths, output_path, grid_size=(2, 2), cell_size=(512, 512)):

    """Create a grid layout from multiple images"""

    grid_width = grid_size[1] * cell_size[0]

    grid_height = grid_size[0] * cell_size[1]

    grid = Image.new('RGB', (grid_width, grid_height), color='white')

    

    max_images = grid_size[0] * grid_size[1]

    for idx, img_path in enumerate(images_paths[:max_images]):

        if os.path.exists(img_path):

            img = Image.open(img_path).resize(cell_size, LANCZOS)

            row = idx // grid_size[1]

            col = idx % grid_size[1]

            grid.paste(img, (col * cell_size[0], row * cell_size[1]))

    

    grid.save(output_path)

    return output_path

def create_comic_strip(images_paths, output_path, orientation='horizontal'):

    """Create a comic strip from multiple images"""

    if not images_paths:

        return None

    

    images = [Image.open(p) for p in images_paths if os.path.exists(p)]

    if not images:

        return None

    

    if orientation == 'horizontal':

        total_width = sum(img.width for img in images)

        max_height = max(img.height for img in images)

        strip = Image.new('RGB', (total_width, max_height), color='white')

        

        x_offset = 0

        for img in images:

            strip.paste(img, (x_offset, 0))

            x_offset += img.width

    else:

                  

        max_width = max(img.width for img in images)

        total_height = sum(img.height for img in images)

        strip = Image.new('RGB', (max_width, total_height), color='white')

        

        y_offset = 0

        for img in images:

            strip.paste(img, (0, y_offset))

            y_offset += img.height

    

    strip.save(output_path)

    return output_path

def check_image_validity(image_path):

    """Check if image is valid and not corrupted"""

    try:

        img = Image.open(image_path)

        img.verify()

        return True

    except:

        return False

def get_image_size(image_path):

    """Get image dimensions"""

    img = Image.open(image_path)

    return img.size

