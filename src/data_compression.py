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
        ## there are arguments you can give it to flatten it differently,
        ## e.g. by columns
        ## I think it should be fine either way (?)

    return img_array_flat

## this function should only need to get called once
## find the mean flattened vector of all the images
def mean_image_vec(path_to_images):
    ## argument should be the directory with all of the processed, binary images
        ## i.e. let's keep all these in their own folder

    all_imgs_list = os.listdir(path_to_images)

    ## get the size of the binary images (they should all be the same size)
        ## this fname string might be off.. 
    dummy_img = cv2.imread(path_to_images+'/'+all_imgs_list[0], 0)

    ## just the 1D size of the processed images
    img_size = np.size(np.array(dummy_img).flatten())
    mean_img = np.zeros(img_size,float)

    for img_name in all_imgs_list:
        full_fname = path_to_images+'/'+img_name
        mean_img += flattener(full_name)

    mean_img /= len(all_imgs_list)
    ## careful, "ints" are going to be rounded off.. and everything is 0 or 1
    ## try having floats in "mean_img" for now

    ## the average flattened image vector of the whole processed dataset:
    return mean_img 

## this function should also only need to get called once
## create the big covariance matrix for the whole processed dataset
    ## see eq'n 8 of the paper
def big_cov_matrix(mean_img,path_to_images):
    ## mean_img should be the output of the "mean_image_vec" function
        ## i.e. the average flattened image vector of the whole dataset
    ## path_to_images should be the folder/directory with all of the processed images

    ## list of all the processed image names, like the above function
    all_imgs_list = os.listdir(path_to_images)

    ## this matrix has columns that are "theta" vectors
        ## num rows = size of these flattened image vectors
        ## num columns = number of images in dataset
    ## dtype also float? 
    A_matrix = np.zeros((np.size(mean_img),len(all_imgs_list)),float)

    for m in range(len(all_imgs_list)):
        ## path to individual (processed) image file
        mth_img_path = path_to_images+'/'+all_imgs_list[m]         
        theta_m = flattener(mth_img_path) - mean_img

        ## might end up needing some "theta_m transpose" on RHS instead? (probably not) 
        A_matrix[:,m] += theta_m

    A_T = np.transpose(A_matrix)

    return np.dot(A_matrix, A_T) ## this should be the covariance matrix
    
    
