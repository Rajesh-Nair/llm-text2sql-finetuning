#!/bin/bash

# Define environment name
ENV_NAME="LM-FT"

# Create a new conda environment with Python 3.10
conda create --name $ENV_NAME python=3.10 -y

# Activate the newly created environment
source activate
conda activate $ENV_NAME

# Install python jupyter kernel
conda install ipykernal
python -m ipykernel install --user --name $ENV_NAME --display-name $ENV_NAME

# Install packages from requirements.txt using pip
pip install -r requirements.txt
