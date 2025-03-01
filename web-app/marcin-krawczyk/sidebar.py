import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import requests

# Streamlit part

st.sidebar.markdown("## About the Pneumonia detection app")
project_description = """The application was created as part of an educational project within the Super Data Science Community
https://community.superdatascience.com.\n 
The aim of the project was to create an application that detects the presence of pneumonia based on an uploaded x-ray image. 
The application was based on multilayer convolutional neural network (CNN) designed for binary image classification. 
It consists of three convolutional layers with ReLU activation, max pooling to reduce spatial dimensions, 
and two fully connected layers, with the final layer using a sigmoid activation for probability output. 
The model was trained using approximately 8,000 lung x-ray images downloaded from  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia and augmented to improve accuracy.\n
The model takes grayscale images as an input and returns a diagnosis along with a probability value. """
st.sidebar.write(project_description)

image = Image.open("web-app/marcin-krawczyk/results.jpg")
st.sidebar.image(image, use_container_width=True)

st.markdown("<h1 style='text-align: center;'>Pneumonia detection</h1>", unsafe_allow_html=True)
