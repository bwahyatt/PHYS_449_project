## define the NN class here

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from nptyping import NDArray
import pandas as pd
from typing import List, Callable
import os
from tqdm import tqdm

import src.data_compression as dc
from src.verbosity_printer import VerbosityPrinter

class GalaxiesDataset(Dataset):
    '''
    Inheritance of the torch.utils.data.Dataset class
    '''
    
    def __init__(self: Dataset, 
                 processed_images_dir: str, 
                 ids_and_labels_path: str, 
                 feature_size: int, 
                 train_dataset: Dataset = None, 
                 vprinter: VerbosityPrinter = None, 
                 transform = None, 
                 target_transform = None) -> None:
        '''
        Inheritance of the torch.utils.data.Dataset class

        Args:
            self (Dataset): An instance of the GalaxiesDataset class
            processed_images_dir (str): Directory to the processed images
            ids_and_labels_path (str): File path to the ids and labels .csv file
            feature_size (int): The number of feature vectors to use to represent the data
            train_dataset (GalaxiesDataset): If we are constructing a training dataset, leave this parameter empty. If we are constructing a testing dataset, pass the training dataset into this parameter so that the testing dataset knows what the mean image and PCA matricies are
            vprinter (VerbosityPrinter): Object that handles verbosity printing
            transform (function, optional): A transformation function to apply to the image data. Defaults to None.
            target_transform (function, optional): A transformation function to apply to the labels. Defaults to None.
        
        Attributes:
            vprinter (VerbosityPrinter): Object that handles verbosity printing
            ids_and_labels_path (str): File path to the ids and labels .csv file
            feature_size (int): The number of feature vectors to use to represent the data
            img_source_path_list (str): List of all images
            mean_vector (NDArray): The mean image of the dataset
            feature_array (NDArray): The features of each image
            class_labels (NDArray): The labels of each image
            transform (function, optional): A transformation function to apply to the image data. Defaults to None.
            target_transform (function, optional): A transformation function to apply to the labels. Defaults to None.
        '''
        if vprinter is None:
            vprinter = VerbosityPrinter() 
        self.vprinter = vprinter
            
        self.processed_images_dir = processed_images_dir 
        self.ids_and_labels_path = ids_and_labels_path
        self.feature_size = feature_size  
        
        # Process and label the data      
        self.img_source_path_list = os.listdir(self.processed_images_dir)
        if train_dataset is None:
            labelled_data = self.compress_and_label_data()
        else:
            labelled_data = self.compress_and_label_data(train_dataset)
        self.mean_vector = labelled_data['mean_vector']
        self.feature_array = labelled_data['feature_array']
        self.class_labels = labelled_data['class_labels']
        
        # Store transformation functions if needed
        self.transform = transform
        self.target_transform = target_transform
        
    def compute_PCA_matrix(self: Dataset, mean_vector: NDArray) -> NDArray:
        '''
        The PCA matrix is huge, 
        so it's best to compute it whenever we need it as opposed to storing it as an attribute

        Args:
            self (GalaxiesDataset): The object representing our dataset of galaxies
            mean_vector (NDArray): The mean image of the dataset

        Returns:
            NDArray: The PCA matrix
        '''
        thetas_mat = dc.matrix_of_thetas(mean_vector, self.processed_images_dir) ## matrix of thetas for whole dataset
        self.vprinter.vprint("theta vector acquired",2)
        PCA_matrix = dc.mat_of_thetas_to_pcs(thetas_mat, self.feature_size)    ## matrix of big C's principle components
        self.vprinter.vprint("PCA matrix acquired",2)
        return PCA_matrix
        
    def compress_and_label_data(self: Dataset, train_dataset: Dataset = None) -> dict:
        '''
        Represents the data using a small feature vector obtained from performing PCA on the dataset
        Then labels all the data using the given .csv file of ids and labels.

        Args:
            self (GalaxiesDataset): The object representing our dataset of galaxies
            train_dataset (GalaxiesDataset): If we are constructing a training dataset, leave this parameter empty. If we are constructing a testing dataset, pass the training dataset into this parameter so that the testing dataset knows what the mean image and PCA matricies are
        Raises:
            ValueError: If we need more classes, an error will be raised

        Returns:
            dict: Contains mean_vector, feature_array, and class_labels
        '''                   
        
        # Calculate the mean, theta, and PCA vectors
        if train_dataset is None:
            mean_vector = dc.mean_image_vec(self.processed_images_dir)
            self.vprinter.vprint("mean vector acquired",2)
            PCA_matrix = self.compute_PCA_matrix(mean_vector)  
        else:
            mean_vector = train_dataset.mean_vector
            PCA_matrix = train_dataset.compute_PCA_matrix(mean_vector)
        
        # Create feature vectors for each image
        processed_imgs_list = self.img_source_path_list
        ids_and_labels = pd.read_csv(self.ids_and_labels_path, index_col='ID')
        feature_array = np.zeros((len(ids_and_labels), self.feature_size), dtype = 'float32') 
        for k in tqdm(range(len(processed_imgs_list)), desc = "Creating feature vectors for each image", disable = self.vprinter.system_verbosity == 0): 
            processed_fname = f'{self.processed_images_dir}/{processed_imgs_list[k]}'
            current_flat_img = dc.flattener(processed_fname)
            current_feature_vec = dc.feature_extract(PCA_matrix, current_flat_img, mean_vector)
            feature_array[k,:] += current_feature_vec
        self.vprinter.vprint('\n', 1)
            
        # And finally, label each image
        class_labels = np.zeros(len(processed_imgs_list), dtype=np.int64) ## according to my (Ben) A2, pytorch is expecting int64 for loss function
        for k in range(np.size(class_labels)):
            
            ## let spiral galaxies have a label = 0
            ## let elliptical galaxies have label = 1
            
            ## WE NEED: some set of INDECIES corresponding to each class, to give as a "label" for our loss function
            ## looks like 'ids_and_labels' just has S and E classes at the moment
            ## this is an ad hoc bit of code for now, should be more generalized in principle
            ## e.g. if the "number of classes" hyperparameter is made > 2
            ## I invite anyone who has a better idea on how to do this to tweak it 
            ## but if we are just doing S and E, need something like:
            
            ## is this how indexing in Pandas works?
            ## i.e. will it recognize that underscore even though the original csv column is "Simple Classification" with a space?
            if ids_and_labels['Simple Classification'][int(processed_imgs_list[k][:-4])] == 'E':
                class_labels[k] = 1
            elif ids_and_labels['Simple Classification'][int(processed_imgs_list[k][:-4])] == 'S':
                continue
            else:
                raise ValueError("your ad hoc label thing needs more classes")
        
        
        labelled_data = {
            'mean_vector': mean_vector,
            'feature_array': feature_array,
            'class_labels': class_labels            
        }
        return labelled_data
        
    def uncompress_and_display_img(self, index: int):
        '''
        Uncompresses and displays the image at index

        Args:
            index (int): The index to use to access a specific image from our dataset
        '''
        dc.uncompress_img(self.compute_PCA_matrix(self.mean_vector), self[index][0], self.mean_vector, display_img = True)
        
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
    
    def __init__(self, feature_dim: int, hidden_nodes: int, num_classes: int):
        super(Net, self).__init__() 
        
        '''
        feature = size of feature vector
            authors use 8, 13, 25
        nodes is the number of hidden nodes
            authors use "one third" of input nodes
        num_classes is number of galaxy classes being considered
            authors use 2, 3, 5, and 7
        '''
        
        self.fc1 = nn.Linear(feature_dim, hidden_nodes)            
        self.fc2 = nn.Linear(hidden_nodes, num_classes)
        
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
    
    def train_model(self, 
              galaxies_data: GalaxiesDataset, 
              loss_fn: Callable, 
              optimizer: Callable, 
              batch_size: int = 1,
              vprinter: VerbosityPrinter = None,
              device: str = 'cpu',
              save_model_path: str = None,
              show_accuracy: bool = False
              ) -> float:
        '''
        Trains the model using the training dataset

        Args:
            galaxies_data (GalaxiesDataset): The training dataset
            loss_fn (Callable): A loss function for penalizing the model
            optimizer (Callable): The optimization function to use to minimize the loss function
            batch_size (int, optional): The number of datapoints to use in each batch. Defaults to 1.
            vprinter (VerbosityPrinter, optional): Object that handles verbosity printing. Defaults to None.
            device (str, optional): The device to use. Defaults to 'cpu'.
            save_model_path (str, optional): The filepath to save model weights to. Defaults to None.
            show_accuracy (bool, optional): If true, will print out the accuracy. Defaults to False.

        Returns:
            float: The loss value after training
        '''
        
        if vprinter is None: 
            vprinter = VerbosityPrinter()
        
        dataloader = DataLoader(galaxies_data, batch_size = batch_size)
        size = len(dataloader.dataset)
        self.train()       
            
        current = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print out progress
            # training loss, training accuracy, test loss, and test accuracy
            lenX = len(X)
            loss = loss.item()
            current += lenX
            
            if show_accuracy:
                correct = (pred.argmax(1) == y).type(torch.float).sum().item()
                training_accuracy = correct / lenX * 100
                vprinter.vprint(f"Training Accuracy: {training_accuracy:.3}% \t Training Loss: {loss:.6f} [{current:>5d}/{size:>5d}]",
                        msg_verbosity = 2)
            
        if save_model_path is not None:
            torch.save(self.state_dict(), save_model_path)
            vprinter.vprint(f'Model weights saved to {save_model_path}',
                            msg_verbosity = 0)
            
        return loss
    
    def test(self,
            galaxies_data: GalaxiesDataset, 
            loss_fn: Callable, 
            vprinter: VerbosityPrinter = None,
            batch_size: int = 1,
            device: str = 'cpu',
            show_accuracy: bool = False
        ) -> float:
        '''
        Tests the model with a testing dataset

        Args:
            galaxies_data (GalaxiesDataset): The training dataset
            loss_fn (Callable): A loss function for penalizing the model
            vprinter (VerbosityPrinter, optional): Object that handles verbosity printing. Defaults to None.
            batch_size (int, optional): The number of datapoints to use in each batch. Defaults to 1.
            device (str, optional): The device to use. Defaults to 'cpu'.
            show_accuracy (bool, optional): If true, will print out the accuracy. Defaults to False.
            
        Returns:
            float: The loss value after training
        '''
        
        dataloader = DataLoader(galaxies_data, batch_size = batch_size)
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        
        if show_accuracy:        
            correct /= size
            vprinter.vprint(f"Test Results: \n Test Accuracy: {(100*correct):>0.1f}% \t Test Loss: {test_loss:>8f} \n",
                    msg_verbosity = 0)
        
        return test_loss
        
    ## copying this from workshop 2 as well
    ## e.g. if we wanna change number of feature vector elements/hidden layers like author, we can use this same outline
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
    
        