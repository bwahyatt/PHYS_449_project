## define the NN class here

from torch import nn
from torch.utils.data import Dataset
import numpy as np 
from nptyping import NDArray
import pandas as pd
from typing import List, Callable
import os

import src.data_compression as dc
from src.verbosity_printer import VerbosityPrinter

class GalaxiesDataset(Dataset):
    '''
    Inheritance of the torch.utils.data.Dataset class
    '''
    
    def __init__(self: Dataset, processed_images_dir: str, ids_and_labels_path: str, N_features: int, vprinter: VerbosityPrinter = None, transform = None, target_transform = None) -> None:
        '''
        Inheritance of the torch.utils.data.Dataset class

        Args:
            self (Dataset): An instance of the GalaxiesDataset class
            processed_images_dir (str): Directory to the processed images
            ids_and_labels_path (str): File path to the ids and labels .csv file
            N_features (int): The number of feature vectors to use to represent the data
            transform (function, optional): A transformation function to apply to the image data. Defaults to None.
            target_transform (function, optional): A transformation function to apply to the labels. Defaults to None.
        '''
        if vprinter is None:
            vprinter = VerbosityPrinter() 
        self.vprinter = vprinter
            
        self.processed_images_dir = processed_images_dir 
        self.ids_and_labels_path = ids_and_labels_path
        self.N_features = N_features  
        
        # Process and label the data      
        self.img_source_path = os.listdir(self.processed_images_dir) 
        labelled_data = self.compress_and_label_data()
        self.mean_vector = labelled_data['mean_vector']
        self.feature_array = labelled_data['feature_array']
        self.class_labels = labelled_data['class_labels']
        
        self.transform = transform
        self.target_transform = target_transform
        
    def compute_PCA_matrix(self: Dataset, mean_vector: NDArray) -> NDArray:
        '''
        The PCA matrix is huge, 
        so it's best to compute it whenever we need it as opposed to storing it as an attribute

        Args:
            self (GalaxiesDataset): The object representing our dataset of galaxies

        Returns:
            NDArray: The PCA matrix
        '''
        thetas_mat = dc.matrix_of_thetas(mean_vector, self.processed_images_dir) ## matrix of thetas for whole dataset
        self.vprinter.vprint("theta vector acquired",2)
        PCA_matrix = dc.mat_of_thetas_to_pcs(thetas_mat, self.N_features)    ## matrix of big C's principle components
        self.vprinter.vprint("PCA matrix acquired",2)
        return PCA_matrix
        
    def compress_and_label_data(self: Dataset) -> dict:
        '''
        Represents the data using a small feature vector obtained from performing PCA on the dataset
        Then labels all the data using the given .csv file of ids and labels.

        Args:
            self (GalaxiesDataset): The object representing our dataset of galaxies
        Raises:
            ValueError: If we need more classes, an error will be raised

        Returns:
            dict: Contains mean_vector, feature_array, and class_labels
        '''                   
        
        # Calculate the mean, theta, and PCA vectors
        mean_vector = dc.mean_image_vec(self.processed_images_dir)
        self.vprinter.vprint("mean vector acquired",2)
        PCA_matrix = self.compute_PCA_matrix(mean_vector)             
        
        # Create feature vectors for each image
        processed_imgs_list = self.img_source_path
        ids_and_labels = pd.read_csv(self.ids_and_labels_path)
        feature_array = np.zeros((len(ids_and_labels), self.N_features), float) 
        for k in range(len(processed_imgs_list)): 
            processed_fname = f'{self.processed_images_dir}/{ids_and_labels.ID[k]}.jpg'
            current_flat_img = dc.flattener(processed_fname)
            current_feature_vec = dc.feature_extract(PCA_matrix, current_flat_img, mean_vector)
            feature_array[k,:] += current_feature_vec
            
        # And finally, label each image
        class_labels = np.zeros((len(ids_and_labels)), dtype=np.int64) ## according to my (Ben) A2, pytorch is expecting int64 for loss function
        for k in range(np.size(class_labels)):
            
            ## let spiral galaxies have a label = 0
            ## let elliptical galaxies have label = 1
            
            ## is this how indexing in Pandas works?
            ## i.e. will it recognize that underscore even though the original csv column is "Simple Classification" with a space?
            if ids_and_labels['Simple Classification'][k] == 'E':
                class_labels[k] = 1
            elif ids_and_labels['Simple Classification'][k] == 'S':
                continue
            else:
                raise ValueError("your ad hoc label thing needs more classes")
        
        
        labelled_data = {
            'mean_vector': mean_vector,
            'feature_array': feature_array,
            'class_labels': class_labels            
        }
        return labelled_data
        
    def __len__(self):
        return len(self.class_labels)
        
    def __getitem__(self, index):
        features = self.feature_array[index]
        label = self.class_labels[index]
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)
        return features, label

class Net(nn.Module):
    
    def __init__(self, feature_dim: int, nodes: int, num_classes: int):
        super(Net, self).__init__() 
        
        '''
        feature = size of feature vector
            authors use 8, 13, 25
        nodes is the number of hidden nodes
            authors use "one third" of input nodes
        num_classes is number of galaxy classes being considered
            authors use 2, 3, 5, and 7
        '''
        
        self.fc1 = nn.Linear(feature_dim, nodes)            
        self.fc2 = nn.Linear(nodes, num_classes)
        
    def forward(self, x):
        ## according to this:
        ## https://www.mathworks.com/help/deeplearning/ref/tansig.html
        ## "tan sigmoid" activation is just tanh?
        ## kind of makes sense, tanh function has similar behaviour to sigmoid, probably just weird '04 terminology
        tan = nn.Tanh()
        h = tan(self.fc1(x)) ## Setting up Tanh and using it on data have to go on separate lines - otherwise an error occurs
        y = self.fc2(h)
        
        ## return y for now? 
        ## if we need e.g. softmax, CrossEntropyLoss will do it for us to this last linear output
        return y
    
    # def train(self, galaxies_data: GalaxiesDataset, loss_fn: Callable, ) -> float:
    
    ## copying this from workshop 2 as well
    ## e.g. if we wanna change number of feature vector elements/hidden layers like author, we can use this same outline
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
    
        