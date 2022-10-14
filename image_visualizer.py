'''
Converts a row in even_mnist.csv into an image
'''

import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import argparse

def show_image(data: list, title = None):
    '''
    Displays the greyscale value of the list on a 14 x 14 grid

    Args:
        data (list): A list with 196 (14 x 14) or 197 elements
        title (str): The title of the plot. (Default: None)
    '''
    
    # Process the data
    if len(data) == 14*14 + 1:
        target = data[-1]
        data = data[:-1]
    else:
        target = None
    data = [int(i) for i in data]
    data = np.reshape(data, (14,14))
    
    # cmap = mpl.colors.Colormap('black')
    plt.imshow(data, cmap='gray', vmin=0, vmax=255)
    if title is not None:
        plt_title = title
    elif target is None:
        plt_title = 'The expected digit is unknown'
    else:
        plt_title = f'This is supposed to be a {str(target)}'
        
    plt.title(plt_title)
    plt.show()
            

def main(): 
    '''
    Default test commands:
    
    cd './Assignment 2'
    python src/image_visualizer.py data/even_mnist.csv
    
    python image_visualizer.py data/even_mnist.csv
    python image_visualizer.py data/even_mnist.csv -s 5
    python image_visualizer.py data/even_mnist.csv -s 20
    '''
    
    # Setup the CLI parser
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help = 'File path to the mnist csv')
    parser.add_argument("-s", '--stop_after', help = 'Stop displaying images after the specified number (default = 10)', default = 10)
    args = parser.parse_args()
    data_path = args.data_path   
    stop_after = args.stop_after
    try: 
        stop_after = int(stop_after)
    except:
        print(f'WARNING: `{stop_after}` could not be converted into an int, defaulting to 10') 
        stop_after = 10
    
    # data_path = 'data/even_mnist.csv'
    # stop_after = 10
    
    i = 1
    with open(data_path, mode = 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        for row in spamreader:
            show_image(row)
            if i >= stop_after:
                break
            i += 1

if __name__ == '__main__':
    main()