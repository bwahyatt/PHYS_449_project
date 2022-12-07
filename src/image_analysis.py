import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import math
from scipy import ndimage, misc
from nptyping import NDArray
from typing import Tuple


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
    if bin_image.max() == 0:
        raise ValueError('Warning, this binary image is entirely blank! Binary threshold may be too high!')
    
    return bin_image

def grayscale_img(path_to_img: str):
    '''
    Produces a greyscale image of a "raw" image from the dataset using OpenCV

    Args:
        path_to_img (str): Filepath to the image
    '''  
    img = cv2.imread(path_to_img)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayscale



def centre_row_col(bin_img_array: NDArray) -> Tuple[int]: 
    '''
    Calculate moments of binary image

    Args:
        bin_img_array (NDArray): The output of "binary_assign" (a 2D numpy array)

    Returns:
        2-Tuple[int]: Indicies of the centre row, column of the galaxy
    '''
    M = cv2.moments(bin_img_array)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

def normalize_binary_image(bin_img_array: NDArray) -> NDArray:
    '''
    Sets all values in the bin_img_array to either 0 or 1

    Args:
        bin_img_array (NDArray): The binary image array with 255 representing the on state

    Returns:
        NDArray: The same binary image array, but with 255 mapped to 1.
    '''
    return bin_img_array / 255

def unnormalize_binary_image(bin_img_array: NDArray) -> NDArray:
    '''
    Inverse mapping of normalize_binary_image()

    Args:
        bin_img_array (NDArray): The binary image array with 1 representing the on state

    Returns:
        NDArray: The same binary image array, but with 1 mapped to 255.
    '''
    return bin_img_array * 255

def small_cov_matrix(bin_img_array: NDArray, centre_i: int, centre_j: int) -> NDArray:
    '''
    Computes the 2x2 cov matrix, per image (equation 4)

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

    Args:
        cov_matrix (NDArray): argument is the 2x2 covariance matrix (np array) output of small_cov_matrix function above

    Returns:
        float: The angle offset of the principal axis of the image from the horizontal
    '''
        
    ## np.linalg.eig(A) returns the eigenvalues and the eigenvectors (in that order) of a square array
    ## eigenvecs are in a 2D array
    ## see this for more info:
    ## https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
    
    C_eigenvals, C_eigenvectors = np.linalg.eig(cov_matrix)
    PC_column = np.argmax(C_eigenvals) ## index of max eigenvalue
    PC1 = C_eigenvectors[:,PC_column]    
        
    theta = np.arctan2(PC1[1],PC1[0])
    return theta                      

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

def rem_back(bin_img, grayscale_img): # Remove background from grayscale image
    bin_img = bin_img.astype(int)
    blk_cols = [] 
    for i in range(bin_img.shape[1]):
        if (np.all(bin_img[:,i] == 0)):
            blk_cols.append(i)
    if (len(blk_cols) == bin_img.shape[1]):
        return bin_img.astype(float)
    else:
        grayscale_img = np.delete(grayscale_img, blk_cols, axis=1)
        return grayscale_img
    
def crop(img, shape):
    crop = cv2.resize(img, dsize = shape)
    return crop 



