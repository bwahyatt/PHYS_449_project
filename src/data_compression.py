import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from nptyping import NDArray
from sklearn.decomposition import PCA

def flattener(path_to_image: str) -> NDArray:
    '''
    Obtains the flattened image vector of the (processed) galaxy image

    Args:
        path_to_image (str): The filepath str to the image to flatten

    Returns:
        NDArray: The flatten image array
    '''

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


def mean_image_vec(path_to_images: str) -> NDArray:
    '''
    Find the mean flattened vector of all the images
    NOTE: This function should only need to get called once!
    
    Args:
        path_to_images (str): The directory with all of the processed, binary images.
            i.e. let's keep all these in their own folder

    Returns:
        NDArray: The average flattened image vector of the whole processed dataset
    '''

    all_imgs_list = os.listdir(path_to_images)

    ## get the size of the binary images (they should all be the same size)
        ## this fname string might be off.. 
    dummy_img = cv2.imread(path_to_images+'/'+all_imgs_list[0], 0)

    ## just the 1D size of the processed images
    img_size = np.size(np.array(dummy_img).flatten())
    mean_img = np.zeros(img_size,float)

    for img_name in all_imgs_list:
        full_fname = path_to_images+'/'+img_name
        mean_img += flattener(full_fname)

    mean_img /= len(all_imgs_list)
    ## careful, "ints" are going to be rounded off.. and everything is 0 or 1
    ## try having floats in "mean_img" for now

    ## the average flattened image vector of the whole processed dataset:
    return mean_img 

def matrix_of_thetas(mean_img: NDArray, path_to_images: str) -> NDArray:
    '''
    Computes how each image differs from the mean image and puts those values together in a matrix

    Args:
        mean_img (NDArray): The output of the "mean_image_vec" function
            i.e. the average flattened image vector of the whole dataset
        path_to_images (str): The folder/directory with all of the processed images

    Returns:
        NDArray: The matrix of thetas
    '''
    
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
        
    return A_matrix

def big_cov_matrix(mean_img: NDArray, path_to_images: str) -> NDArray:
    '''
    Create the big covariance matrix for the whole processed dataset
        - See eq'n 8 of the paper
    NOTE: This function should also only need to get called once
    
    Args:
        mean_img (NDArray): The output of the "mean_image_vec" function
            i.e. the average flattened image vector of the whole dataset
        path_to_images (str): The folder/directory with all of the processed images

    Returns:
        NDArray: The covariance matrix
    '''

    # ## list of all the processed image names, like the above function
    # all_imgs_list = os.listdir(path_to_images)

    # ## this matrix has columns that are "theta" vectors
    #     ## num rows = size of these flattened image vectors
    #     ## num columns = number of images in dataset
    # ## dtype also float? 
    # A_matrix = np.zeros((np.size(mean_img),len(all_imgs_list)),float)

    # for m in range(len(all_imgs_list)):
    #     ## path to individual (processed) image file
    #     mth_img_path = path_to_images+'/'+all_imgs_list[m]         
    #     theta_m = flattener(mth_img_path) - mean_img

    #     ## might end up needing some "theta_m transpose" on RHS instead? (probably not) 
    #     A_matrix[:,m] += theta_m

    A_matrix = matrix_of_thetas(mean_img, path_to_images)
    A_T = np.transpose(A_matrix)

    return np.dot(A_matrix, A_T) ## this should be the covariance matrix
    
def cov_to_pcs(cov_mat: NDArray, n_components: int) -> NDArray:
    '''
    Returns the first n eigenvalues of the covariant matrix. 
    First n is decided by taking the largest n eigenvalues' corresponding eigenvectors

    Args:
        cov_mat (NDArray): The big covariant matrix

    Returns:
        NDArray: A 2D-Array, each column is an eigenvector of cov_mat
    '''

    pca = PCA(n_components=n_components)
    result = pca.fit_transform(cov_mat)

    return result

## STILL NEED:
    ## PCA on the big cov matrix above (i.e. get set of eigenvectors ranked by eigenvalues)
    ## feature extraction (projecting theta vectors onto the^ principle components of C)