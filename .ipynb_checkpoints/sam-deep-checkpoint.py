"""
weights downloaded from https://github.com/facebookresearch/segment-anything 
"""

import os 
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import rescale
import seaborn as sns
import skimage.metrics
from skimage import io
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
sys.path.append("..")
import cv2
import os.path
from PIL import Image

def set_matplotlib_font():
    font_families = matplotlib.rcParams['font.sans-serif']
    if font_families[0] != 'Arial':
        font_families.insert(0, 'Arial')
    matplotlib.rcParams['pdf.fonttype'] = 42

def show_anns(anns, ax):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

if __name__ == '__main__':
    set_matplotlib_font()
    pdf_file = "hubble_patches.pdf"
    image_dir = "hubble_patches"
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    tile_size = 512

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator_ = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.96,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    pdf_pages = PdfPages(pdf_file)

    count = 0

    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            print(f"{filename} - {count}")
            count += 1
            if count >= 15:
                break

            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)

            num_tiles_x = image.shape[1] // tile_size
            num_tiles_y = image.shape[0] // tile_size

            for y in range(num_tiles_y):
                for x in range(num_tiles_x):
                    left = x * tile_size
                    top = y * tile_size
                    right = left + tile_size
                    bottom = top + tile_size
                    tile = image[top:bottom, left:right]
                    masks = mask_generator_.generate(tile)

                    plt.figure(figsize=(10, 10))
                    plt.imshow(tile)
                    show_anns(masks, plt.gca())
                    plt.axis('off')
                    plt.title(filename)
                    pdf_pages.savefig()
                    plt.close()

    pdf_pages.close()
    print("Done")
