import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

## produce a binary image of a "raw" image from the dataset
def binary_assign(path_to_img, threshold):
    ## path_to_img is the folder/file/pathway of the "raw" dataset image
    ## threshold is the lower boundary cut off value for assigning a 1 or 0 
        ## a hyperparameter? Or is there a more systematic way of obtaining it?
        
        #### AT SOME POINT:
        ## we may want to also introduce an upper limit threshold value e.g. for bright foreground stars contaminating images
        ## this is not included in the original paper
    
    ## assign binary pixel values to an image, given a threshold value
    img_input = cv2.imread(path_to_img, 0)
        ## the 0 argument reads the image as black and white
        ## => 2D array, no RGB channels, easy

    img_array = np.array(img_input)
    
    ## using the same i,j,m,n conventions as the paper
    m = np.size(img_array[:,0]) ## number of rows
    n = np.size(img_array[0,:]) ## number of columns
    
    ## CAREFUL with this datatype later, Pytorch is very picky about it
    bin_image = np.zeros(np.shape(img_array),int)
    
    for i in range(m):
        for j in range(n):
            if img_array[i,j] > threshold:
                bin_image[i,j] += 1
                
    ## return the np array
    return bin_image
    
## get the centre row and centre column of a galaxy WITHIN its image
    ## i.e. not necessarily the IMAGE'S centre row/column
    ## need this to define 2x2 covariance matrix, see below
def centre_row_col(bin_img_array): #.astype(int)): ## <-- this astype might be redundant if you use "int" in the binary_assign function
    ## argument is the output of "binary_assign" (a 2D numpy array)
        ## need it with int values here because the centre row, column values are indecies of the array
        ## i.e. can't be floats
        
    ## using the same i,j,m,n conventions as the paper
    m = np.size(bin_img_array[:,0]) ## number of rows
    n = np.size(bin_img_array[0,:]) ## number of columns
    
    centre_i = 0 ## centre row index
    centre_j = 0 ## centre column index
    
    for i in range(m):
        for j in range(n):
            centre_i += i*bin_img_array[i,j]
            centre_j += j*bin_img_array[i,j]
            
    centre_i /= (m*n)
    centre_j /= (m*n)
    
    ## these should be ints
    return centre_i, centre_j

## gets the 2x2 cov matrix, per image (equation 4)
    ## not to be confused with the big covariance matrix, see "data_compression.py"
def small_cov_matrix(bin_img_array, centre_i, centre_j):
    ## bin_img_array is the 2D array of a binary image of a galaxy
        ## output of "binary_assign" function
    ## centre_i and centre_j are the row and column index (respectively) 
        ## output of "centre_row_col" function
        
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

## this returns the angle to rotate the image by to align the galaxy's main axis along the horizontal
    ## not to be confused with the flattened theta "vectors". see "data_compression.py"/paper section 2.2
## SIGN CONVENTION?
    ## I **think** if theta > 0, rotate the image clockwise by theta
    ## if theta < 0, rotate the image counterclockwise by theta
def theta_angle(cov_matrix):
    ## argument is the 2x2 covariance matrix (np array)
        ## output of small_cov_matrix function above
        
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
        
    theta = np.arctan2(PC1[1],PC1[0]) ## again, might need to swap those two arguments later
    return theta

## STILL NEEDED:
    ## an actual image rotater (by the angle theta, obtained above) 
    ## something that re-scales the images/gets rid of background columns
        ## authors are somewhat ambiguous on how they did it..
    ## something that saves np array as an image file
        

    
    
