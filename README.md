# PHYS_449_project

Repository for our PHYS 449 group project

## Dataset format

Our dataset should have a folder with all the images, and a txt/csv file with these rows:

- object name (i.e. NGC No.),
- full classification (look up on ned.ipac site),
- "binary" classication,
- "trinary" classification, etc.

I.e. the authors use multiple numbers of classification outputs. Keep the "full" morphology output so we can go back and decide which/how many classifications we want for the other rows

## Project Dependencies

### Python

We are using Python 3.10.5

### Create a new venv (virtual environment)

```sh
python -m venv group_proj_env
```

Note: you could also use conda, or another package manager to create and manage virtual environments.

### Activate a venv

1. Using VSCode's file explorer, go to `group_proj_env/Scripts/`
2. Right click on the file named `activate`(should be the first one)
3. Click on "Copy Relative Path"
4. Paste this path in the command line and hit enter

### Packages Required

- numpy
- opencv
- matplotlib
- pandas
- requests
- scikit-learn
- pytorch

### Use this command to install the required packages

```sh
pip install -r requirements.txt
```
