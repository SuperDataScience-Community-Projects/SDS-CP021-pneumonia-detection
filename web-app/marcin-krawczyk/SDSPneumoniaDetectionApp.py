# libraries
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import requests


# parameters
xysize = 128
predictin_threshold = 0.7
# MODEL_URL = "https://drive.google.com/file/d/1uc6dMUxBSNqAAXLZzLBy0GG7vUAOzNGS/view?usp=drive_link"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def download_file(drive_link, output_path="downloaded_file"):
#     file_id = drive_link.split("/d/")[1].split("/")[0]  # Pobranie ID pliku z linku
#     url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
#     response = requests.get(url, stream=True)
#     if response.status_code == 200:
#         with open(output_path, "wb") as file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 file.write(chunk)
#         return output_path
#     else:
#         print("Nie udało się pobrać pliku.")
#         return None

# CNN Model definition
class BasicCNNV2(nn.Module):
    def __init__(self):
        super(BasicCNNV2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * (xysize // 8) * (xysize // 8), out_features=512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load model
model = BasicCNNV2()
model.load_state_dict(torch.load("web-app/marcin-krawczyk/model128px25e.pth", map_location=device))
# model_weights_path = download_file(MODEL_URL)
# state_dict = torch.load(model_weights_path, map_location=torch.device("cpu"), weights_only=True)
# model.load_state_dict(state_dict)
model.to(device)
model.eval()


# Definition of X-rey transformation
transform_image = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((xysize, xysize)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Function to prepare transformed image for visualization
def process_image_for_visualization(image):
    image = image.squeeze(0)  
    image = image.numpy()  # to numpy conversion
    image = (image * 0.5) + 0.5  # denormalization
    image = np.clip(image, 0, 1)  
    return image

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

uploaded_file = st.file_uploader("Upload the X-rey photo (JPEG)", type=["jpg", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    # original photo visualization
    st.image(original_image, caption="Uploade X-rey photo", use_container_width=True)

    # Przetwarzanie obrazu
    image_tensor = transform_image(original_image).unsqueeze(0).to(device)
    
    # Predykcja
    with torch.no_grad():
        output = model(image_tensor).item()
    
    probability = output
    if probability > predictin_threshold:
#        result = "Probable pneumonia"
        # st.error("##Probable pneumonia##")
        # st.write(f"### :red[Probable pneumonia {probability*100:.1f}%]")
        st.markdown(f"<h3 style='text-align: center; color: red;'>Pneumonia with probability {probability*100:.1f}%</h3>", unsafe_allow_html=True)

    else:
        st.markdown(f"<h3 style='text-align: center; color: green;'>No pneumonia with probability {(1-probability)*100:.1f}%</h3>", unsafe_allow_html=True)





