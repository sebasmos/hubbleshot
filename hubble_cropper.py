from PIL import Image
import os
import tifffile
import numpy as np

# Create the output directory if it doesn't exist
output_dir = "hubble_patches"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Determine the image format
FORMAT = "tiff"

# Load the image
if FORMAT == "tiff":
    img_path = "hubble.tif"
    image = tifffile.imread(img_path)
    image = np.array(image)
    print(f"Shape is: {image.shape}, min is {image.min()} and max is {image.max()}")
    image_size = image.shape[:2]  # Extract the width and height
else:
    image_size = (1000, 1000)
    image = Image.new("RGB", image_size)
    image.save("hubble_1000x1000.png")

# Convert the image array to PIL Image
image = Image.fromarray(image)
# Define the size of the patches
patch_size = (200, 200)

# Calculate the number of patches in each dimension
num_patches_x = image_size[0] // patch_size[0]
num_patches_y = image_size[1] // patch_size[1]

# Crop and save the patches
for i in range(num_patches_x):
    for j in range(num_patches_y):
        left = i * patch_size[0]
        upper = j * patch_size[1]
        right = left + patch_size[0]
        lower = upper + patch_size[1]
        patch = image.crop((left, upper, right, lower))
        patch.save(f"{output_dir}/hubble_patch_{i}_{j}.png")
