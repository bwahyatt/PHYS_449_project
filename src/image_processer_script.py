import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

## this is currently just an OUTLINE of what an image processing script could/should look like
    ## this shouldn't be done in "main.py" since in principle we only need to do this once
## i.e. it is not currently runnable, but highlights how some of these functions from src folder could fit together
## should go through raw images and save the binary

# sys.path.append('src')

from image_analysis import binary_assign, centre_row_col, small_cov_matrix, theta_angle, rotate

## DATASET STUFF...
## we should have some folder like "dataset" with all of our raw images in it 
    ## see "raw_images"
## might want another script that saves images to a folder, e.g. with that sdss package
    ## see "dataset_generate.py"

## we should make a separate folder with all of the processed, binary images too (see below)

binary_threshold = 123 ## some number for assigning binary image pixel values
                       ## could also be included in some hyperparameter dictionary like the assignments, or command line argument?
                       ## or obtained some other clever way (automation, etc)
                         ## potentially: opencv tools (see "image_analysis.py", last few lines) 
        
        
PATH_TO_RAW_IMAGES = "../raw_images/"    ## relative should work if this script is in 'src'
PATH_TO_PROCESSED_IMAGES = "../processed_images/"    ## "  "  "  "  "  "
        
raw_img_names = os.listdir(PATH_TO_RAW_IMAGES) ## list of the raw image file names

for n in range(len(raw_img_names)):
    binary_image_array = binary_assign(PATH_TO_RAW_IMAGES+raw_img_names[n], binary_threshold)
    i_bar, j_bar = centre_row_col(binary_image_array)
    C_2x2 = small_cov_matrix(binary_image_array, i_bar, j_bar)
    theta_rotate_angle = theta_angle(C_2x2)
    rotated_image_array = rotate(binary_image_array, theta_rotate_angle) ## this takes radian angle argument (I think)
    
    ## rotate, rescale, save array as a processed image to new folder
    
    