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
import os
import numpy as np
from skimage import io
import matplotlib
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import os.path
from PIL import Image

markers = ['Hoechst','AF1','CD31','CD45','CD68','Blank','CD4',
              'FOXP3','CD8a','CD45RO','CD20','PD-L1','CD3e','CD163',
              'E-Cadherin','PD-1','Ki-67','Pan-CK','SMA']

def plot_imgs(imgs, titles):
    """
    Generate visualization of list of arrays
    :param imgs: list of arrays, each numpy array is an image of size (width, height)
    :param titles: list of titles [string]
    """
    # create figure
    fig = plt.figure(figsize=(10, 4))
    # loop over images
    for i in range(len(imgs)):
        fig.add_subplot(3, 7, i + 1)
        plt.imshow(imgs[i])
        plt.title(str(titles[i]), fontsize=8)
        plt.axis("off")
    return fig


def return_list_of_matches(key, elements):
    """
    uses the key (identifier for each generated image) and 
    returns on a list all the elemets on the path that contain that key.
    In general should be Real A, Fake B, Real B
    """
    lista_elementos = []
    for i in elements:
        if key in i:
            lista_elementos.append(i)
    return lista_elementos### Watch results

def set_matplotlib_font():
    font_families = matplotlib.rcParams['font.sans-serif']
    if font_families[0] != 'Arial':
        font_families.insert(0, 'Arial')
    matplotlib.rcParams['pdf.fonttype'] = 42
def compare(img1, img2, img_he=None):

    fig = plt.figure( figsize=(10, 4))
    _fig1, _fig2 = fig.subfigures(1, 2, wspace=2, width_ratios=[1, 5])

    if img_he is not None:
        axs1 = _fig1.subplots(3, 1)
        #axs1[0].imshow(rescale(img_he, (1,1, 1)), alpha=0.5)
        axs1[0].imshow(img_he)
        axs1[1].imshow(np.dstack([img1[1], img1[1], np.zeros_like(img1[1])]), alpha=0.5)
        
        axs1[2].imshow(np.dstack([img2[2], img2[2], np.zeros_like(img1[2])]), alpha=0.5)
        for ax in axs1.flat:
            ax.axis('off')
            ax.set_title('')

    axs2 = _fig2.subplots(4, 10)
    
    mask_generator_ = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.96,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    
    for i, ax, m in zip(img1, axs2.flat[::2], markers):
        ax.imshow(i, vmin=0, vmax=255)
        ax.set_title(m)
    for i, ax in zip(img2, axs2.flat[1::2]):
        ax.imshow(i, vmin=0, vmax=255)
    for i, j, ax in zip(img1, img2, axs2.flat[1::2]):
        from sklearn.metrics.cluster import normalized_mutual_info_score
        ax.set_title(f"{normalized_mutual_info_score(i.flatten(), j.flatten()):.04f}",  fontsize=8)
    for ax in axs2.flat:
        ax.axis('off')
    return fig


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
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
    pdf_file = "hubble_250x250.pdf"
    path_images = '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/scajas/repositories/vcg/3DStyleGAN/hubble_patches'
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    tile_size = 200

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator_ = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.96,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    pdf_pages = PdfPages(pdf_file)
 
    count = 0      

    for filename in os.listdir(path_images):    
        if filename.endswith('.png'):
            # count+=1
            # if count >=2:
            #     break
            image_path = os.path.join(path_images, filename)
            image = cv2.imread(image_path)

            num_tiles_x = image.shape[1] // tile_size
            num_tiles_y = image.shape[0] // tile_size

            for y in range(num_tiles_y):
                for x in range(num_tiles_x):
                        # print(y,x, num_tiles_y, num_tiles_x)
                        left = x * tile_size
                        top = y * tile_size
                        right = left + tile_size
                        bottom = top + tile_size
                        # Crop the image to this region
                        tile = image[top:bottom, left:right]
                        masks = mask_generator_.generate(tile)
                        plt.figure(figsize=(10,10))
                        plt.imshow(tile)
                        show_anns(masks)
                        plt.axis('off')
                        pdf_pages.savefig() 
                        plt.close()  


    pdf_pages.close()  # close the pdf file
    print("Done")