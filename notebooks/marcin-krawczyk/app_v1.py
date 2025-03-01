import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Definicja modelu (musi być identyczna jak wcześniej)
class BasicCNNV2(nn.Module):
    def __init__(self, xysize=64):  # Dodajemy xysize jako argument
        super(BasicCNNV2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Head of the CNN
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


# Funkcja do wczytywania modelu (cache'owana)
@st.cache_resource
def load_model():
    model = BasicCNNV2(xysize=64)  # Ustaw rozmiar zgodnie z trenowanym modelem
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()
st.write("Model załadowany!")


dummy_input = torch.tensor(np.random.rand(1, 1, 64, 64).astype(np.float32))
prediction = model(dummy_input).item()
st.write(f"Wynik predykcji: {prediction:.4f}")

