import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple
from tqdm import tqdm

## this is currently just an OUTLINE of what an image processing script could/should look like
    ## this shouldn't be done in "main.py" since in principle we only need to do this once
## i.e. it is not currently runnable, but highlights how some of these functions from src folder could fit together
## should go through raw images and save the binary

# sys.path.append('src')

from image_analysis import binary_assign, centre_row_col, small_cov_matrix
from image_analysis import theta_angle, rotate, crop, normalize_binary_image

## DATASET STUFF...
## we should have some folder like "dataset" with all of our raw images in it 
    ## see "raw_images"
## might want another script that saves images to a folder, e.g. with that sdss package
    ## see "dataset_generate.py"

## we should make a separate folder with all of the processed, binary images too (see below)
    ## see "processed_images"

binary_threshold = 123 ## some number for assigning binary image pixel values
                       ## could also be included in some hyperparameter dictionary like the assignments, or command line argument?
                       ## or obtained some other clever way (automation, etc)
                         ## potentially: opencv tools (see "image_analysis.py", last few lines) 
final_shape = (128, 128)      
   
def import_raw_imgs(pdir: str) -> Tuple[str]:
    PATH_TO_RAW_IMAGES = f"{pdir}/raw_images/"    ## relative should work if this script is in 'src'
    PATH_TO_PROCESSED_IMAGES = f"{pdir}/processed_images/"    ## "  "  "  "  "  "
    raw_img_names = os.listdir(PATH_TO_RAW_IMAGES) 
    return PATH_TO_RAW_IMAGES, PATH_TO_PROCESSED_IMAGES, raw_img_names
    
try:
    pdir = '..'
    PATH_TO_RAW_IMAGES, PATH_TO_PROCESSED_IMAGES, raw_img_names = import_raw_imgs(pdir)
except:
    pdir = '.'
    PATH_TO_RAW_IMAGES, PATH_TO_PROCESSED_IMAGES, raw_img_names = import_raw_imgs(pdir)

## raw_img_names = list of the raw image file names

for n in tqdm(range(len(raw_img_names)), desc = 'Processing Images'):
    binary_image_array = binary_assign(PATH_TO_RAW_IMAGES+raw_img_names[n], binary_threshold)
    normd_bin_img_arr = normalize_binary_image(binary_image_array)
    i_bar, j_bar = centre_row_col(normd_bin_img_arr)
    C_2x2 = small_cov_matrix(normd_bin_img_arr, i_bar, j_bar)
    theta_rotate_angle = theta_angle(C_2x2)
    rotated_image_array = rotate(normd_bin_img_arr, theta_rotate_angle) ## this takes radian angle argument (I think)
    processed_image = crop(rotated_image_array, final_shape)
    
    ## dropping the '.jpg' from the string, looks like cv2 takes care of it
    ## our processed images end with .jpg.jpg
    cv2.imwrite(f'../processed_images/{raw_img_names[n]}', processed_image)
    ## rotate, rescale, save array as a processed image to new folder
    
    