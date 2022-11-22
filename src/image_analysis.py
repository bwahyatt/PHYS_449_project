import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import math
from scipy import ndimage, misc
from nptyping import NDArray
from typing import Tuple

# def binary_assign(path_to_img: str, threshold: int, output_path: str) -> NDArray:
#     '''
#     Produce a binary image of a "raw" image from the dataset

#     Args:
#         path_to_img (str): The folder/file/pathway of the "raw" dataset image
#         threshold (int): The lower boundary cut off value for assigning a 1 or 0 (choose a number between 0 and 255)
#             a hyperparameter? Or is there a more systematic way of obtaining it?
#         output_path (str): The filepath of where we want to save the image to

#     Returns:
#         NDArray: The array of 1s and 0s instead of 255s and 0s in case the fact that white is 1 and not 255 is important later on

#     '''
        
#     #### AT SOME POINT:
#     ## we may want to also introduce an upper limit threshold value e.g. for bright foreground stars contaminating images
#     ## this is not included in the original paper
    
#     ## assign binary pixel values to an image, given a threshold value
#     img_input = cv2.imread(path_to_img, 0)
#         ## the 0 argument reads the image as black and white
#         ## => 2D array, no RGB channels, easy

#     img_array = np.array(img_input)
    
#     ## using the same i,j,m,n conventions as the paper
#     m = np.size(img_array[:,0]) ## number of rows
#     n = np.size(img_array[0,:]) ## number of columns
    
#     ## CAREFUL with this datatype later, Pytorch is very picky about it
#     bin_image = np.zeros(np.shape(img_array),int)
    
#     for i in range(m):
#         for j in range(n):
#             if img_array[i,j] > threshold:
#                 bin_image[i,j] += 1
#     # Multiply the array by 255 because cv2.imwrite takes the array values as colour values
#     # And we want the pixels above the threshold to appear white
#     im_array = 255*bin_image

#     cv2.imwrite(output_path, im_array)
#     ## this^ intermediate, unrotated binary image will need to be used later in "rotator" function, see below

#     ## return the np array
#     # returning the array of 1s and 0s instead of 255s and 0s in case the fact that white is 1 and not 255 is important later on
#     return bin_image

def binary_assign(path_to_img: str, threshold: int) -> NDArray:
    '''
    Produce a binary image of a "raw" image from the dataset using OpenCV

    Args:
        path_to_img (str): Filepath to the image
        threshold (int): The lower bound of the pixel value that we will replace with a 1. Any values lower than the threshold will be replaced with 0

    Returns:
        NDArray: The binary image array
    '''
    img = cv2.imread(path_to_img)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_val = 255
    ret, bin_image = cv2.threshold(grayscale, threshold, max_val, cv2.THRESH_BINARY)  
    return bin_image


# def centre_row_col(bin_img_array: NDArray) -> Tuple[int]: #.astype(int)): ## <-- this astype might be redundant if you use "int" in the binary_assign function   
#     '''
#     Get the centre row and centre column of a galaxy WITHIN its image
#         ## i.e. not necessarily the IMAGE'S centre row/column
#         ## need this to define 2x2 covariance matrix, see below

#     Args:
#         bin_img_array (NDArray): The output of "binary_assign" (a 2D numpy array)
#             ## need it with int values here because the centre row, column values are indicies of the array
#             ## i.e. can't be floats
#     Returns:
#         2-Tuple[int]: Indicies of the centre row, column of the galaxy
#     '''
        
#     ## using the same i,j,m,n conventions as the paper
#     m = np.size(bin_img_array[:,0]) ## number of rows
#     n = np.size(bin_img_array[0,:]) ## number of columns
    
#     centre_i = 0 ## centre row index
#     centre_j = 0 ## centre column index
    
#     for i in range(m):
#         for j in range(n):
#             centre_i += i*bin_img_array[i,j]
#             centre_j += j*bin_img_array[i,j]
            
#     centre_i /= (m*n)
#     centre_j /= (m*n)
    
#     ## these should be ints
#     return centre_i, centre_j

def centre_row_col(bin_img_array: NDArray) -> Tuple[int]: 
    '''
    Calculate moments of binary image

    Args:
        bin_img_array (NDArray): The output of "binary_assign" (a 2D numpy array)

    Returns:
        2-Tuple[int]: Indicies of the centre row, column of the galaxy
    '''
    M = cv2.moments(bin_img_array)
    # calculate x,y coordinate of center
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def small_cov_matrix(bin_img_array: NDArray, centre_i: int, centre_j: int) -> NDArray:
    '''
    Computes the 2x2 cov matrix, per image (equation 4)
    ## not to be confused with the big covariance matrix, see "data_compression.py"

    Args:
        bin_img_array (NDArray): The 2D array of a binary image of a galaxy output of "binary_assign" function
        centre_i (int): The centre row index. Output of centre_row_col()
        centre_j (int): The centre column index. Output of centre_row_col()

    Returns:
        NDArray: The 2x2 cov matrix
    '''
        
    C_matrix = np.zeros((2,2),int)
    
    m = np.size(bin_img_array[:,0]) ## number of rows
    n = np.size(bin_img_array[0,:]) ## number of columns 
    
    for i in range(m):
        for j in range(n):
            C_matrix[0,0] += bin_img_array[i,j]*((i-centre_i)**2)
            C_matrix[0,1] += bin_img_array[i,j]*(i-centre_i)*(j-centre_j)
            C_matrix[1,0] += bin_img_array[i,j]*(i-centre_i)*(j-centre_j)
            C_matrix[1,1] += bin_img_array[i,j]*((j-centre_j)**2)
            
    return C_matrix


def theta_angle(cov_matrix: NDArray) -> float:
    '''
    This returns the angle to rotate the image by to align the galaxy's main axis along the horizontal
        ## not to be confused with the flattened theta "vectors". see "data_compression.py"/paper section 2.2
    ## SIGN CONVENTION?
        ## I **think** if theta > 0, rotate the image clockwise by theta
        ## if theta < 0, rotate the image counterclockwise by theta

    Args:
        cov_matrix (NDArray): argument is the 2x2 covariance matrix (np array) output of small_cov_matrix function above

    Returns:
        float: The angle offset of the principal axis of the image from the horizontal
    '''
        
    ## np.linalg.eig(A) returns the eigenvalues and the eigenvectors (in that order) of a square array
        ## eigenvals are in a 1D array
        ## eigenvecs are in a 2D array "such that the COLUMN v[:,i] is the eigenvector corresponding to the eigenvalue w[i]."
    
    ## see this for more info:
    ## https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
    
    C_eigenvals, C_eigenvectors = np.linalg.eig(cov_matrix)
    PC_column = np.argmax(C_eigenvals) ## index of max eigenvalue (this can only be 0 or 1 b/c it's a 2x2 matrix)
    PC1 = C_eigenvectors[:,PC_column]    
    
    #### OKAY, I (Ben) am not certain about all of these details below
    ## "PC1" (assigned above) is what the paper calls the C matrix eigenvector with the max eigenvalue
    ## this vector is 2D
    ## the components of PC1 define the angle of the galaxy's main axis WRT the horizontal (I am taking the authors' word on that)
    ## ambiguous: which components of PC1 in equation 5 are the "x" component vs "y" component?
        ## for now, I am going to assume PC1[0] is horizontal/x component, PC1[1] is vertical/y component
        ## NOTE ALSO: the author's use indecies 1 and 2 rather than 0 and 1
    ## also: below I am using arctan2 (which returns an angle from -pi to pi rather than -pi/2 to pi/2, like typical arctan)
        ## I believe (?) this is the correct choice
        
    theta = np.arctan2(PC1[1],PC1[0])  ## again, might need to swap those two arguments later
    return theta                       ## again, this is in radians, from -pi to pi

def rotate(image: NDArray, angle: float) -> NDArray:
    '''
    Rotates an image by an angle

    Args:
        image (NDArray): The image to rotate
        angle (float): The angle to rotate the image by

    Returns:
        NDArray: The rotated image
    '''
    angle_deg = math.degrees(angle)
    output = ndimage.rotate(image, angle_deg, reshape=False)
    return output

## STILL NEEDED:
    ## an actual image rotater (by the angle theta, obtained above) 
    ## something that re-scales the images/gets rid of background columns
        ## authors are somewhat ambiguous on how they did it..
    ## something that saves np array as an image file

## This could work for rotation:
## https://pyimagesearch.com/2021/01/20/opencv-rotate-image/
## https://www.geeksforgeeks.org/python-opencv-getrotationmatrix2d-function/


# if __name__ == '__main__':

    ## Keeping the stuff using cv2.threshold down here
    ## In case we want to replace something in the current binary_assign function with it
    ## Otherwise we can delete it


    # gal = cv2.imread('../raw_images/1237661949186474088.jpg',0) #Import a raw image as greyscale

    # # converting to its binary form
    # (thresh,binary_array) = cv2.threshold(gal, 90, 255, cv2.THRESH_BINARY)

    # cv2.imwrite('./test_binary_image.jpg', binary_array)
    # print(binary_array)

    # binary_assign('../raw_images/1237651753493266487.jpg', 100, '../data//test_binary.jpg') #Creating a test image


