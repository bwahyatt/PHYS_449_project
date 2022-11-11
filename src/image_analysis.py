import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def binary_assign(path_to_img, threshold):
    
    ## assign binary pixel values to an image, given a threshold value
    img_input = cv2.imread(path_to_img, 0)
        ## the 0 argument reads the image as black and white
            ## => 2D array, no RGB channels, easy

    img_array = np.array(img_input)
    
    ## using the same i,j,m,n conventions as the paper
    m = np.size(img_array[:,0]) ## number of rows
    n = np.size(img_array[0,:]) ## number of columns
    
    ## careful with this datatype later, Pytorch is very picky about it
    bin_image = np.zeros(np.shape(img_array),float)
    
    for i in range(m):
        for j in range(n):
            if img_array[m,n] > threshold:
                bin_image[m,n] += 1.0
                
    ## return the np array?
    return bin_image
    
def centre_row_col(bin_img_array.astype(int)):
    ## argument is the output of "binary_assign"
        ## need it with int values here because the centre row, column values are indecies of the array
        ## i.e. can't be floats
    ## using the same i,j,m,n conventions as the paper
    m = np.size(bin_img_array[:,0]) ## number of rows
    n = np.size(bin_img_array[0,:]) ## number of columns
    
    centre_i = 0
    centre_j = 0
    
    for i in range(m):
        for j in range(n):
            centre_i += i*bin_img_array[m,n]
            centre_j += j*bin_img_array[m,n]
            
    centre_i /= (m*n)
    centre_j /= (m*n)
    
    return centre_i, centre_j
    
    
    
    
