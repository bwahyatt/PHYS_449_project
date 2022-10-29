# PHYS_449_project

Repository for our PHYS 449 group project

## Dataset format

Our dataset should have a folder with all the images, and a txt/csv file with these rows:

** object name (i.e. NGC No.), full classification (look up on ned.ipac site), "binary" classication, "trinary" classification, etc ** 

I.e. the authors use multiple numbers of classification outputs. Keep the "full" morphology output so we can go back and decide which/how many classifications we want for the other rows

## `image_visualizer.py`

This script requires the following packages:

```{cmd}
pip install numpy matplotlib 
```

To run this script, enter the following into the command line:

```{cmd}
python image_visualizer.py data/even_mnist.csv
```
