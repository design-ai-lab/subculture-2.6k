
import sys as _sys
from pathlib import Path as _Path
for _p in _Path(__file__).resolve().parents:
    if (_p / "paths.py").is_file():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break
from paths import DATASET, CODE_ROOT, OUTPUT, under
for _d in (
    CODE_ROOT / "results" / "laion_profile",
    CODE_ROOT / "results" / "separability",
    CODE_ROOT / "results" / "matched_text",
    CODE_ROOT / "dataset",
    CODE_ROOT / "scoring",
):
    if _d.is_dir() and str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import os
import json
from PIL import Image
import plotly.graph_objects as go
from colorthief import ColorThief
import webcolors
import numpy as np
from sklearn.cluster import KMeans
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color, sRGBColor

"""
This script analyzes the dominant colors of the corpus images, one subculture folder at a time, and visualizes the results.

How it works:
- Each image is resized to 150x150 and its pixels are clustered into 5 groups using KMeans on RGB values
  (random_state=42). CIE LAB is used only for color naming and distances, not for the clustering.
- It counts how many pixels belong to each cluster.
- It calculates the percentage of the image that each color represents, based on actual pixel counts.
- It outputs these percentages and color information to a JSON file and creates a visualization.

What the percentages mean:
- The percentages are the true proportions of each dominant color in the image, not arbitrary or fixed values.
- The color grouping is based on clustering similar colors, so visually similar shades are grouped together.

Usage:
    python dataset/analyze_colors.py                      # all six subcultures
    python dataset/analyze_colors.py --subculture y2k

Outputs (output/color_analysis/):
- <subculture>.json: color analysis per image, same layout as data/color_analysis/<subculture>.json.
- top_colors_<subculture>.html: interactive visualization of the top 3 colors for each image.

Palettes can differ slightly from the released files under other scikit-learn versions.
"""

# Hardcoded CSS3 color hex-to-name mapping for compatibility
CSS3_HEX_TO_NAMES = {
    '#f0f8ff': 'aliceblue', '#faebd7': 'antiquewhite', '#00ffff': 'aqua', '#7fffd4': 'aquamarine',
    '#f0ffff': 'azure', '#f5f5dc': 'beige', '#ffe4c4': 'bisque', '#000000': 'black', '#ffebcd': 'blanchedalmond',
    '#0000ff': 'blue', '#8a2be2': 'blueviolet', '#a52a2a': 'brown', '#deb887': 'burlywood', '#5f9ea0': 'cadetblue',
    '#7fff00': 'chartreuse', '#d2691e': 'chocolate', '#ff7f50': 'coral', '#6495ed': 'cornflowerblue', '#fff8dc': 'cornsilk',
    '#dc143c': 'crimson', '#00ffff': 'cyan', '#00008b': 'darkblue', '#008b8b': 'darkcyan', '#b8860b': 'darkgoldenrod',
    '#a9a9a9': 'darkgray', '#006400': 'darkgreen', '#bdb76b': 'darkkhaki', '#8b008b': 'darkmagenta', '#556b2f': 'darkolivegreen',
    '#ff8c00': 'darkorange', '#9932cc': 'darkorchid', '#8b0000': 'darkred', '#e9967a': 'darksalmon', '#8fbc8f': 'darkseagreen',
    '#483d8b': 'darkslateblue', '#2f4f4f': 'darkslategray', '#00ced1': 'darkturquoise', '#9400d3': 'darkviolet', '#ff1493': 'deeppink',
    '#00bfff': 'deepskyblue', '#696969': 'dimgray', '#1e90ff': 'dodgerblue', '#b22222': 'firebrick', '#fffaf0': 'floralwhite',
    '#228b22': 'forestgreen', '#ff00ff': 'fuchsia', '#dcdcdc': 'gainsboro', '#f8f8ff': 'ghostwhite', '#ffd700': 'gold',
    '#daa520': 'goldenrod', '#808080': 'gray', '#008000': 'green', '#adff2f': 'greenyellow', '#f0fff0': 'honeydew',
    '#ff69b4': 'hotpink', '#cd5c5c': 'indianred', '#4b0082': 'indigo', '#fffff0': 'ivory', '#f0e68c': 'khaki',
    '#e6e6fa': 'lavender', '#fff0f5': 'lavenderblush', '#7cfc00': 'lawngreen', '#fffacd': 'lemonchiffon', '#add8e6': 'lightblue',
    '#f08080': 'lightcoral', '#e0ffff': 'lightcyan', '#fafad2': 'lightgoldenrodyellow', '#d3d3d3': 'lightgray', '#90ee90': 'lightgreen',
    '#ffb6c1': 'lightpink', '#ffa07a': 'lightsalmon', '#20b2aa': 'lightseagreen', '#87cefa': 'lightskyblue', '#778899': 'lightslategray',
    '#b0c4de': 'lightsteelblue', '#ffffe0': 'lightyellow', '#00ff00': 'lime', '#32cd32': 'limegreen', '#faf0e6': 'linen',
    '#ff00ff': 'magenta', '#800000': 'maroon', '#66cdaa': 'mediumaquamarine', '#0000cd': 'mediumblue', '#ba55d3': 'mediumorchid',
    '#9370db': 'mediumpurple', '#3cb371': 'mediumseagreen', '#7b68ee': 'mediumslateblue', '#00fa9a': 'mediumspringgreen', '#48d1cc': 'mediumturquoise',
    '#c71585': 'mediumvioletred', '#191970': 'midnightblue', '#f5fffa': 'mintcream', '#ffe4e1': 'mistyrose', '#ffe4b5': 'moccasin',
    '#ffdead': 'navajowhite', '#000080': 'navy', '#fdf5e6': 'oldlace', '#808000': 'olive', '#6b8e23': 'olivedrab',
    '#ffa500': 'orange', '#ff4500': 'orangered', '#da70d6': 'orchid', '#eee8aa': 'palegoldenrod', '#98fb98': 'palegreen',
    '#afeeee': 'paleturquoise', '#db7093': 'palevioletred', '#ffefd5': 'papayawhip', '#ffdab9': 'peachpuff', '#cd853f': 'peru',
    '#ffc0cb': 'pink', '#dda0dd': 'plum', '#b0e0e6': 'powderblue', '#800080': 'purple', '#ff0000': 'red',
    '#bc8f8f': 'rosybrown', '#4169e1': 'royalblue', '#8b4513': 'saddlebrown', '#fa8072': 'salmon', '#f4a460': 'sandybrown',
    '#2e8b57': 'seagreen', '#fff5ee': 'seashell', '#a0522d': 'sienna', '#c0c0c0': 'silver', '#87ceeb': 'skyblue',
    '#6a5acd': 'slateblue', '#708090': 'slategray', '#fffafa': 'snow', '#00ff7f': 'springgreen', '#4682b4': 'steelblue',
    '#d2b48c': 'tan', '#008080': 'teal', '#d8bfd8': 'thistle', '#ff6347': 'tomato', '#40e0d0': 'turquoise',
    '#ee82ee': 'violet', '#f5deb3': 'wheat', '#ffffff': 'white', '#f5f5f5': 'whitesmoke', '#ffff00': 'yellow', '#9acd32': 'yellowgreen'
}

def closest_color(requested_color):
    min_colors = {}
    for hex_value, name in CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def rgb_to_lab(rgb):
    """Convert RGB to LAB color space"""
    rgb_color = sRGBColor(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    return convert_color(rgb_color, LabColor)

def color_distance(rgb1, rgb2):
    """Calculate color distance in LAB space"""
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    return delta_e_cie2000(lab1, lab2)

def get_color_name(rgb_color):
    try:
        return webcolors.rgb_to_name(rgb_color)
    except Exception:
        return closest_color(rgb_color)

def analyze_image_colors(image_path):
    # Open image and convert to RGB
    img = Image.open(image_path)
    img = img.convert('RGB')
    
    # Resize image for faster processing while maintaining color distribution
    # For a 150x150 image, we'll have 22,500 pixels to analyze
    img = img.resize((150, 150))
    
    # Convert image to numpy array
    # Each pixel is represented as an RGB value (3 numbers: Red, Green, Blue)
    # For a 150x150 image, pixels.shape will be (22500, 3)
    pixels = np.array(img)
    pixels = pixels.reshape(-1, 3)
    
    # Use KMeans to find dominant colors
    # KMeans is an unsupervised machine learning algorithm that groups similar colors together
    # n_clusters=5: We want to find 5 dominant colors in the image
    # random_state=42: For reproducibility (same results each time)
    kmeans = KMeans(n_clusters=5, random_state=42)
    
    # The clustering process:
    # 1. Initialize: Randomly pick 5 points in RGB space as initial "centers"
    # 2. Assignment: For each pixel, find the closest center and assign it to that cluster
    # 3. Update: Move each center to the average position of all pixels in its cluster
    # 4. Repeat steps 2 and 3 until centers stop moving
    kmeans.fit(pixels)
    
    # Get colors and their counts
    colors = []
    # unique_labels: Which cluster each pixel belongs to (0-4)
    # counts: How many pixels are in each cluster
    unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
    total_pixels = int(sum(counts))  # Convert to Python int
    
    # Sort colors by frequency
    # color_counts: List of (color_center, pixel_count) pairs
    # Sort by count in descending order to get most dominant colors first
    color_counts = list(zip(kmeans.cluster_centers_, counts))
    color_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Convert colors to RGB and calculate percentages
    for rgb, count in color_counts:
        # Convert float RGB values to integers (0-255)
        rgb = rgb.astype(int)
        # Convert RGB to hex color code (e.g., #FF0000 for red)
        hex_color = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
        # Get the closest named color (e.g., "red", "blue", etc.)
        color_name = get_color_name(tuple(rgb))
        # Calculate what percentage of the image this color represents
        percentage = float((count / total_pixels) * 100)  # Convert to Python float
        
        colors.append({
            'rgb': tuple(map(int, rgb)),  # Convert numpy.int64 to Python int
            'hex_color': hex_color,
            'color_name': color_name,
            'percentage': percentage
        })
    
    return colors

def analyze_folder(folder_path):
    image_analysis = []
    
    # Get all jpg files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        colors = analyze_image_colors(image_path)
        
        image_analysis.append({
            'filename': image_file,
            'colors': colors
        })
    
    return {'image_analysis': image_analysis}

SUBCULTURES = ["cottagecore", "cyberpunk", "dark_academia", "goblincore", "synthwave", "y2k"]


def get_output_names(subculture):
    """Output directory and file names for one subculture."""
    output_dir = OUTPUT / "color_analysis"
    json_file = output_dir / f"{subculture}.json"
    html_file = output_dir / f"top_colors_{subculture}.html"
    return output_dir, json_file, html_file

def plot_top_colors(data, output_html, top_n=3):
    # Create a figure with subplots for each image
    fig = go.Figure()

    # Add color strips for each image
    for i, image in enumerate(data['image_analysis']):
        top_colors = sorted(image['colors'], key=lambda x: x['percentage'], reverse=True)[:top_n]
        for j, color in enumerate(top_colors):
            fig.add_trace(go.Bar(
                x=[image['filename']],
                y=[color['percentage']],
                marker_color=color['hex_color'],
                name=f"{color['color_name']}",
                showlegend=True
            ))

    # Update layout for full webpage size
    fig.update_layout(
        title=f'Top {top_n} Colors for Each Image',
        xaxis_title='Image',
        yaxis_title='Percentage',
        height=800,
        width=1200,
        barmode='stack'
    )

    # Save the plot as an HTML file
    fig.write_html(output_html)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="dominant-color analysis of the corpus images")
    ap.add_argument("--subculture", choices=SUBCULTURES, action="append",
                    help="subculture folder to analyze (repeatable; default: all six)")
    args = ap.parse_args()

    for subculture in args.subculture or SUBCULTURES:
        folder_path = DATASET / "data" / "images" / subculture
        output_dir, json_file, html_file = get_output_names(subculture)
        os.makedirs(output_dir, exist_ok=True)

        data = analyze_folder(folder_path)
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        plot_top_colors(data, html_file)
        print(f"[done] {subculture}: {len(data['image_analysis'])} images -> {under(OUTPUT, json_file)}")

if __name__ == '__main__':
    main() 