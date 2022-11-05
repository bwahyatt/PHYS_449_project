import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def flattener(path_to_image):
    ## obtains flattened image vector
    ## of (processed) galaxy image

    img_input = cv2.imread(path_to_image, 0)
        ## the 0 argument reads the image as black and white
            ## => 2D array, no RGB channels (?)
        ## might end up being redundant due to binary image processing
    img_array = np.array(img_input)
    #cv2.imshow("display",img_array)

    img_array_flat = img_array.flatten()
    ## note: np.ndarray.flatten (by default) will basically concatenate
    ## the array's ROWS together, from top to bottom

    #print(np.shape(img_array_flat))
    
    return img_array_flat

def mean_image_vec(path_to_images):
    ## find the mean flattened vector of all the images
    ## argument should be the directory with all of the processed, binary images
        ## i.e. let's keep all these in their own folder

    all_imgs_list = os.listdir(path_to_images)

    ## get the size of the binary images (they should all be the same size)
        ## this fname string might be off.. 
    dummy_img = cv2.imread(path_to_images+'/'+all_imgs_list[0], 0)

    ## just the 1D size of the processed images
    img_size = np.size(np.array(dummy_img).flatten())
    mean_img = np.zeros(img_size,int)

    for img_name in all_imgs_list:
        full_fname = path_to_images+'/'+img_name
        mean_img += flattener(full_name)

    mean_img /= len(all_imgs_list)
    ## careful, "ints" are going to be rounded off.. and everything is 0 or 1

    ## the average flattened image vector of the whole processed dataset:
    return mean_img 

        

