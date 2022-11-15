import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

## this is currently just an OUTLINE of what an image processing script could/should look like
    ## this shouldn't be done in "main.py" since in principle we only need to do this once
## i.e. it is not currently runnable, but highlights how some of these functions from src folder could fit together
## should go through raw images and save the binary

sys.path.append('src')

from image_analysis import binary_assign, centre_row_col, small_cov_matrix, theta_angle

## DATASET STUFF...
## we should have some folder like "dataset" with all of our raw images in it 
## might want another script that saves images to a folder, e.g. with that sdss package
## or, we save the images ourselves
## we should make a separate folder with all of the processed, binary images too (see below)

binary_threshold = 123 ## some number for assigning binary image pixel values
                       ## could also be included in some hyperparameter dictionary like the assignments, or command line argument?
                       ## or obtained some other clever way (automation, etc)
        
PATH_TO_RAW_IMAGES = "./TBD" ## ./ might be unneccesary
PATH_TO_PROCESSED_IMAGES = "./TBD2"
        
raw_img_names = os.listdir(PATH_TO_RAW_IMAGES) ## list of the file names

for n in range(len(raw_img_names)):
    binary_image_array = binary_assign(PATH_TO_RAW_IMAGES+"/"+raw_img_names[n], binary_threshold)
    i_bar, j_bar = centre_row_col(binary_image)
    C_2x2 = small_cov_matrix(binary_image_array, i_bar, j_bar)
    theta_rotate = theta_angle(C_2x2)
    
    ## rotate, rescale, save array as a processed image to new folder
    
    