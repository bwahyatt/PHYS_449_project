## define the NN class here

import torch
from torch import nn

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
    
    ## copying this from workshop 2 as well
    ## e.g. if we wanna change number of feature vector elements/hidden layers like author, we can use this same outline
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
    
        