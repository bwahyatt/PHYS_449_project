import cv2
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
from nptyping import NDArray
from sklearn.decomposition import PCA
from typing import Tuple
from datetime import datetime as dt
import scipy as sp

import sys
sys.path.append('src')
from image_analysis import normalize_binary_image, unnormalize_binary_image
from verbosity_printer import VerbosityPrinter
from remove_rogue_files import list_dir

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
    img_array = np.array(img_input).astype(np.float32)
    #cv2.imshow("display",img_array)

    img_array_flat = img_array.flatten()
    ## note: np.ndarray.flatten (by default) will basically concatenate
    ## the array's ROWS together, from top to bottom
        ## there are arguments you can give it to flatten it differently,
        ## e.g. by columns
        ## I think it should be fine either way (?)
        
    ## adding the binary normalization here
    return normalize_binary_image(img_array_flat)

def unflattener(img_arr: NDArray, new_shape: Tuple = (128,128)) -> NDArray:
    '''
    Converts the image array back into a 128 x 128 image array

    Args:
        img_arr (NDArray): The flattened image array
        new_shape (Tuple, optional): The dimensions of the new array. Defaults to (128,128).

    Returns:
        NDArray: The unflattened array
    '''
    return np.reshape(img_arr, new_shape)

def show_img_arr(img_arr: NDArray, output_path: str = None, show_img: bool = True, title: str = None, unflatten: bool = False):
    '''_summary_

    Args:
        img_arr (NDArray): An image array
        output_path (str, optional): Path to output the plot file to. Defaults to None, in which case, no file is outputted.
        show_img (bool, optional): Indicates whether to show the image. Defaults to True.
        title (str, optional): The title for the image. Defaults to None, in which case, no title is given.
    '''
    if unflatten:
        img_arr = unflattener(img_arr)
    plt.imshow(img_arr, cmap = 'gray')
    if title is not None:
        plt.title(title)
    if output_path is not None:
        plt.imsave(output_path, img_arr, cmap = 'gray')
        if not show_img:
            plt.close()
    if show_img:
        plt.show()

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

    all_imgs_list = list_dir(path_to_images, '.DS_Store')
    if all_imgs_list[0] == '.DS_Store':
        all_imgs_list.pop(0)

    ## get the size of the binary images (they should all be the same size)
        ## this fname string might be off.. 
    
    dummy_img = cv2.imread(path_to_images+'/'+all_imgs_list[0], 0)

    ## just the 1D size of the processed images
    img_size = np.size(np.array(dummy_img).flatten())
    mean_img = np.zeros(img_size,float)

    for img_name in all_imgs_list:
        if img_name == '.DS_Store':
            continue
        full_fname = path_to_images+'/'+img_name
        
        ## added the binary normalization in flattener
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
    all_imgs_list = list_dir(path_to_images, '.DS_Store')

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

    A_matrix = matrix_of_thetas(mean_img, path_to_images)
    A_T = np.transpose(A_matrix)

    return np.dot(A_matrix, A_T) ## this should be the covariance matrix
    
def mat_of_thetas_to_pcs(mat_of_thetas: NDArray, n_components: int, method: str = 'sklearn', vprinter: VerbosityPrinter = None, **kwargs) -> NDArray:
    '''
    Returns the first n eigenvalues of the covariant matrix. 
    First n is decided by taking the largest n eigenvalues' corresponding eigenvectors

    Args:
        mat_of_thetas (NDArray): The output of matrix_of_thetas()

    Returns:
        NDArray: A 2D-Array, each column is an eigenvector of mat_of_thetas
    '''
    if vprinter is None:
        vprinter = VerbosityPrinter()   

    start_time = dt.now()
    if method == 'sklearn':
        pca = PCA(n_components=n_components, **kwargs)
        result = pca.fit_transform(mat_of_thetas)
        for i in range(result.shape[1]):
            result[:,i] /= la.norm(result[:,i])
            
    elif method == 'scipy.linalg':
        mat_of_thetas = sp.sparse.bsr_matrix(mat_of_thetas)
        C = mat_of_thetas @ mat_of_thetas.T
        w, result = sp.sparse.linalg.eigsh(C, k = n_components)
    else:
        raise ValueError('Unrecognized `method`. `method` should be one of "sklearn" or "numpy.linalg"')

    end_time = dt.now()
    vprinter.vprint(f"{method} execution time = {end_time-start_time} sec", 1)
    return result

## see: equation 9 of paper
def feature_extract(pca_matrix: NDArray, flat_img: NDArray, mean_img: NDArray) -> NDArray:
    '''
    Args (all arrays):
        pca_matrix = output of mat_of_thetas_to_pcs function  (2D)
        flat_img = flattened vector of a processed image, output of flattener function (1D)
        mean_img = averaged flattened image vector of whole processed dataset, output of mean_image_vec function (1D)
            (in whatever script we end up running with these, it is going to be best to call this mean function globally a single time
            e.g. rather than calling it inside this function, which has to be called for every processed image)
    
    Returns:
        1D array of an image's feature vector
    '''
    
    PCsT = np.transpose(pca_matrix)
    proj = np.dot(PCsT, flat_img-mean_img)
    #proj /= la.norm(proj)
    return proj
    
def uncompress_img(PC_mat: NDArray, feature_vec: NDArray, mean_img_arr: NDArray, display_img: bool = False) -> NDArray:
    '''
    Returns the uncompressed version of the image from it's feature vector and the PC_mat that compressed it.

    Args:
        PC_mat (NDArray): The matrix that was used to compress the image
        feature_vec (NDArray): The compressed features of the image
        mean_img_arr (NDArray): The mean image array
        display_img (bool, optional): Indicates whether the resulting image is displayed. Defaults to False.

    Returns:
        NDArray: The uncompressed image
    '''
    img_arr = np.matmul(PC_mat, feature_vec)
    img_arr = (img_arr) / (np.abs(img_arr).max()) + (mean_img_arr)
    img_arr = unflattener(img_arr)
    if display_img:
        show_img_arr(img_arr, title = f'Number of PCs = {feature_vec.shape[0]}')
    return img_arr

def save_eigengalaxies(PC_mat: NDArray, output_dir: str):
    '''
    Saves all the eigengalaxies stored in PC_mat to the output_dir

    Args:
        PC_mat (NDArray): The matrix of eigengalaxies
        output_dir (str): The directory to output all the images
    '''
    n_vecs = PC_mat.shape[1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(n_vecs):
        show_img_arr(unflattener(PC_mat[:,i]), f'{output_dir}/eigengalaxy_{i}.png', False)

if __name__ == '__main__':
    
    proc_path = os.path.abspath('./processed_images/train')
    feature_size = 8
    mean_vector = mean_image_vec(proc_path)   
    thetas_mat = matrix_of_thetas(mean_vector, proc_path) 
    PCA_matrix = mat_of_thetas_to_pcs(thetas_mat, feature_size, 'sklearn')
    
    # mean_vector = mean_image_vec(proc_path)   
    # processed_fname = f'{proc_path}/1237661968495935570.jpg'
    # current_flat_img = flattener(processed_fname)
    # current_feature_vec = feature_extract(PCA_matrix, current_flat_img, mean_vector)
    # uncomp_img = uncompress_img(PCA_matrix, current_feature_vec, mean_vector, display_img=True)
    
    out_path = 'sandbox/outputs'
    save_eigengalaxies(PCA_matrix, out_path)