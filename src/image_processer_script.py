import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple
from tqdm import tqdm
import json
from verbosity_printer import VerbosityPrinter


import sys
sys.path.append('src')

from image_analysis import binary_assign, centre_row_col, small_cov_matrix, grayscale_img, rem_back
from image_analysis import theta_angle, rotate, crop, normalize_binary_image, unnormalize_binary_image
from remove_rogue_files import list_dir

def import_raw_imgs(pdir: str) -> Tuple[str]:
    PATH_TO_RAW_IMAGES = f"{pdir}/raw_images/"   
    PATH_TO_PROCESSED_IMAGES = f"{pdir}/grayscale_images/"
    raw_img_names = list_dir(PATH_TO_RAW_IMAGES, '.DS_Store')
    return PATH_TO_RAW_IMAGES, PATH_TO_PROCESSED_IMAGES, raw_img_names

def main():
    with open('param/param.json') as fp:
        params = json.load(fp)
    binary_threshold = params['optim']['binary_threshold'] 
    final_shape = (128, 128)      
    
        
    try:
        pdir = '..'
        PATH_TO_RAW_IMAGES, PATH_TO_PROCESSED_IMAGES, raw_img_names = import_raw_imgs(pdir)
    except:
        pdir = '.'
        PATH_TO_RAW_IMAGES, PATH_TO_PROCESSED_IMAGES, raw_img_names = import_raw_imgs(pdir) ## raw_img_names = list of the raw image file names
    

    vprinter = VerbosityPrinter(1)
    for n in tqdm(range(len(raw_img_names)), desc = 'Processing Images', disable = vprinter.system_verbosity == 0):
        # Import image as a grayscale array and normalize it
        
        bin_image_array = binary_assign(PATH_TO_RAW_IMAGES+raw_img_names[n], binary_threshold)
        gray_image_array = grayscale_img(PATH_TO_RAW_IMAGES+raw_img_names[n])
        normd_gray_img_arr = normalize_binary_image(gray_image_array) #the normalize_binary_image function works for grayscale too
        normd_bin_image_array = normalize_binary_image(bin_image_array)

        # Perform the various image processing steps
        i_bar, j_bar = centre_row_col(normd_bin_image_array)
        C_2x2 = small_cov_matrix(normd_bin_image_array, i_bar, j_bar)
        theta_rotate_angle = theta_angle(C_2x2)
        rotated_grayscale_image_array = rotate(normd_gray_img_arr, theta_rotate_angle) ## this takes radian angle argument
        rotated_bin_image_array = rotate(normd_bin_image_array, theta_rotate_angle)
        rem_cols_grayscale = rem_back(rotated_bin_image_array, rotated_grayscale_image_array)
        processed_image = crop(rem_cols_grayscale, final_shape)
        
        # Unnormalize the processed image
        processed_image = unnormalize_binary_image(processed_image)
        # Save array as a processed image to new folder
        cv2.imwrite(f'{pdir}/grayscale_images/{raw_img_names[n]}', processed_image)

if __name__ == '__main__':
    main()
    
    
