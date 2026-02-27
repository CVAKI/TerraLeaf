import os
import sys

# Block TensorFlow from loading if installed
sys.modules['tensorflow'] = None

import pandas as pd
from PIL import Image

# Load CSV data
data = pd.read_csv('data.csv')
data = data[:1000]

# Load all images into imgdata dict { 'img_0001.png': <PIL Image>, ... }
img_folder = 'leaf_images'   # change to your image folder path

imgdata = {}
for i in range(1, 1001):
    filename = f'img_{i:04d}.png'
    filepath = os.path.join(img_folder, filename)
    if os.path.exists(filepath):
        imgdata[filename] = Image.open(filepath).copy()  # .copy() so file handle is released
    else:
        print(f'Warning: {filename} not found')

print(f'Total images loaded: {len(imgdata)}')
print(data.head())