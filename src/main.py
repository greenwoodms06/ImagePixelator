# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 20:47:35 2025

@author: Scott
"""

from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
import numpy as np
from pathlib import Path


def create_directory(directory_path):
    """
    Checks if a directory exists and creates it recursively if it doesn't.

    Args:
        directory_path (str or Path): The path to the directory.
    """
    path = Path(directory_path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{path}' created or already exists.")
    except OSError as e:
        print(f"Error creating directory '{path}': {e}")
        
def pixelate_image(image_path, rows, columns, output_path="image_pixelated.png", grid_color=None, max_colors=10):
    """
    Pixelates an image, preserves appearance through smart color selection,
    adds a grid, and reduces the color palette to a specified maximum size.

    Args:
        image_path (str): Path to the input image file.
        rows (int): Number of rows for pixelation.
        columns (int): Number of columns for pixelation.
        output_path (str, optional): Path to save the pixelated image.
                                     Defaults to "pixelated_image_limited_colors.png".
        grid_color (tuple, str, optional): Color of the grid lines.
                                            Defaults to light gray (192, 192, 192).
        max_colors (int, optional): Maximum number of distinct colors to use in the pixelated image.
                                     Defaults to 10. Set to 0 or None to disable color reduction.
    """
    try:
        img = Image.open(image_path).convert("RGB") # Ensure RGB even if input is RGBA or grayscale
        width, height = img.size

        # Calculate block dimensions
        block_width = width // columns
        block_height = height // rows

        # Resize to exact grid dimensions
        resized_width = block_width * columns
        resized_height = block_height * rows
        img = img.resize((resized_width, resized_height))
        width, height = resized_width, resized_height

        pixelated_img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(pixelated_img)

        block_colors_list = [] # Store block colors for palette reduction

        for i in range(rows):
            for j in range(columns):
                # Block boundaries
                left = j * block_width
                upper = i * block_height
                right = (j + 1) * block_width
                lower = (i + 1) * block_height

                block = img.crop((left, upper, right, lower))

                # Smart Color Selection (Average Color - as before)
                colors = block.getcolors(block_width * block_height)
                if colors is None or len(colors) > (block_width * block_height) // 2:
                    r_sum, g_sum, b_sum = 0, 0, 0
                    pixel_count = 0
                    for x in range(block_width):
                        for y in range(block_height):
                            pixel = block.getpixel((x, y))
                            if isinstance(pixel, tuple) and len(pixel) >= 3:
                                r_sum += pixel[0]
                                g_sum += pixel[1]
                                b_sum += pixel[2]
                                pixel_count += 1
                    if pixel_count > 0:
                        avg_r = int(r_sum / pixel_count)
                        avg_g = int(g_sum / pixel_count)
                        avg_b = int(b_sum / pixel_count)
                        block_color = (avg_r, avg_g, avg_b)
                    else:
                        block_color = (128, 128, 128)
                else:
                    if colors:
                        most_frequent_color_tuple = max(colors, key=lambda item: item[0])
                        block_color = most_frequent_color_tuple[1]
                        if not isinstance(block_color, tuple) or len(block_color) < 3:
                            block_color = (128, 128, 128)
                    else:
                        block_color = (128, 128, 128)

                block_colors_list.append(block_color) # Add block color to the list

                pixel_block = Image.new("RGB", (block_width, block_height), block_color)
                pixelated_img.paste(pixel_block, (left, upper))

                # Draw Grid Lines
                if grid_color:
                    if j < columns - 1:
                        draw.line([(right - 1, upper), (right - 1, lower -1)], fill=grid_color, width=1)
                    if i < rows - 1:
                        draw.line([(left, lower - 1), (right - 1, lower - 1)], fill=grid_color, width=1)

        # Color Palette Reduction (if max_colors > 0)
        if max_colors and max_colors > 0:
            unique_block_colors = list(set(block_colors_list)) # Get unique colors
            if len(unique_block_colors) > max_colors:
                kmeans = KMeans(n_clusters=max_colors, random_state=0, n_init=10) # n_init for stability
                color_array = np.array(unique_block_colors) / 255.0  # Normalize for KMeans (0-1 range)
                kmeans.fit(color_array)
                reduced_palette_normalized = kmeans.cluster_centers_
                reduced_palette = [tuple(int(c * 255) for c in color) for color in reduced_palette_normalized] # Back to 0-255
                
                # Create a color mapping dictionary
                color_map = {}
                for original_color in unique_block_colors:
                    min_distance = float('inf')
                    closest_color = None
                    for palette_color in reduced_palette:
                        distance = sum([(a - b) ** 2 for a, b in zip(original_color, palette_color)]) # Squared Euclidean distance
                        if distance < min_distance:
                            min_distance = distance
                            closest_color = palette_color
                    color_map[original_color] = closest_color

                # Re-color the pixelated image using the reduced palette
                pixelated_img_reduced_colors = Image.new("RGB", (width, height))
                draw_reduced = ImageDraw.Draw(pixelated_img_reduced_colors) # New draw object for new image

                for i in range(rows):
                    for j in range(columns):
                        left = j * block_width
                        upper = i * block_height
                        block_index = i * columns + j
                        original_block_color = block_colors_list[block_index]
                        closest_palette_color = color_map[original_block_color]
                        pixel_block_reduced = Image.new("RGB", (block_width, block_height), closest_palette_color)
                        pixelated_img_reduced_colors.paste(pixel_block_reduced, (left, upper))
                        if grid_color: # Redraw grid on the reduced color image
                            if j < columns - 1:
                                draw_reduced.line([(j * block_width + block_width - 1, i * block_height), (j * block_width + block_width - 1, (i + 1) * block_height -1)], fill=grid_color, width=1)
                            if i < rows - 1:
                                draw_reduced.line([(j * block_width, i * block_height + block_height - 1), ((j + 1) * block_width - 1, i * block_height + block_height - 1)], fill=grid_color, width=1)

                pixelated_img_reduced_colors.save(output_path)
                print(f"Pixelated image with grid and reduced color palette (max {max_colors} colors) saved to {output_path}")
                return # Exit here after saving reduced color image

        # If no color reduction or if less colors than max_colors, save original pixelated image
        pixelated_img.save(output_path)
        print(f"Pixelated image with grid saved to {output_path}")


    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    image_path = Path('../resources/mountain.jpg')
    # image_path = Path('../resources/pendant.png')
    rows = 86
    columns = 64
    output_path = Path('../temp/') / Path(image_path.stem + '.png')
    grid_color = (192, 192, 192)
    max_colors=10
    create_directory(output_path.parent)
    pixelate_image(image_path, rows, columns, output_path, grid_color, max_colors)
    
    # image_file = input("Enter the path to your image file: ")
    # try:
    #     num_rows = int(input("Enter the number of rows for pixelation: "))
    #     num_columns = int(input("Enter the number of columns for pixelation: "))
    #     if num_rows <= 0 or num_columns <= 0:
    #         print("Error: Rows and columns must be positive integers.")
    #     else:
    #         output_file = input("Enter the desired output file name (e.g., pixelated_limited.png, or leave blank for default 'pixelated_image_limited_colors.png'): ")
    #         if not output_file:
    #             output_file = "pixelated_image_limited_colors.png"

    #         grid_color_input = input("Enter grid color (e.g., 'black', 'white', or RGB tuple like '(255,0,0)', or leave blank for no grid): ")
    #         grid_color = None
    #         if grid_color_input:
    #             try:
    #                 if grid_color_input.startswith('(') and grid_color_input.endswith(')'):
    #                     grid_color = eval(grid_color_input)
    #                     if not isinstance(grid_color, tuple) or len(grid_color) != 3:
    #                         raise ValueError
    #                 else:
    #                     grid_color = grid_color_input
    #             except:
    #                 print("Warning: Invalid grid color format. Using default light gray grid.")
    #                 grid_color = (192, 192, 192)
    #         else:
    #             print("No grid color specified, no grid will be drawn.")

    #         max_colors_input = input("Enter maximum number of colors to use (integer, e.g., 10, or leave blank for no color reduction): ")
    #         max_colors = 0 # Default to no color reduction
    #         if max_colors_input:
    #             try:
    #                 max_colors = int(max_colors_input)
    #                 if max_colors < 0:
    #                     max_colors = 0 # Treat negative as no reduction
    #             except ValueError:
    #                 print("Warning: Invalid max colors value. No color reduction will be applied.")
    #                 max_colors = 0

    #         pixelate_image(image_file, num_rows, num_columns, output_file, grid_color, max_colors)
    # except ValueError:
    #     print("Error: Please enter valid integer values for rows and columns.")