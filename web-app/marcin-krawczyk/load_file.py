# libraries
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np



# # CNN Model definition
# class BasicCNNV2(nn.Module):
#     def __init__(self, xysize=64):  # Dodajemy xysize jako argument
#         super(BasicCNNV2, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.relu = nn.ReLU()

#         # Head of the CNN
#         self.fc1 = nn.Linear(128 * (xysize // 8) * (xysize // 8), out_features=512)
#         self.fc2 = nn.Linear(512, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = self.pool(self.relu(self.conv3(x)))
#         x = x.view(x.size(0), -1) 
#         x = self.relu(self.fc1(x))
#         x = self.sigmoid(self.fc2(x))
#         return x 

# Definition of X-rey transformation
transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),  # Ustawiony na 64 zgodnie z modelem
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Funkcja do przekształcenia obrazu i konwersji do formatu wyświetlanego w Streamlit
def process_image(image):
    image = transform_test(image)  # Przekształcenie obrazu
    image = image.squeeze(0)  # Usunięcie kanału batch
    image = image.numpy()  # Konwersja na NumPy
    image = (image * 0.5) + 0.5  # Denormalizacja (przywrócenie wartości do zakresu 0-1)
    image = np.clip(image, 0, 1)  # Upewnienie się, że wartości są w zakresie [0,1]
    return image

# Interfejs Streamlit
st.title("Predykcja zdjęcia rentgenowskiego")
uploaded_file = st.file_uploader("Wgraj zdjęcie rentgenowskie (JPEG)", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Wczytaj i wyświetl oryginalne zdjęcie
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Oryginalne zdjęcie", use_container_width=True)

    # Przetwarzanie obrazu
    processed_image = process_image(original_image)

    # Wyświetlenie przekształconego obrazu
    # st.image(processed_image, caption="Przekształcone zdjęcie", use_column_width=True, clamp=True)
