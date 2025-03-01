import os
import streamlit as st

# Pobranie bieżącego katalogu
current_dir = os.getcwd()
st.write("Current working directory:", current_dir)

# Sprawdzenie, czy model istnieje
MODEL_PATH = "./model.pth"
if os.path.exists(MODEL_PATH):
    st.success("Model file found!")
else:
    st.error("Model file NOT found! Please check file path.")

# Pobranie listy plików i katalogów
st.write("### List of files and directories:")
entries = os.listdir(current_dir)
for entry in entries:
    full_path = os.path.join(current_dir, entry)
    if os.path.isdir(full_path):
        st.write(f"📁 {entry}/")  # Katalog
    else:
        st.write(f"📄 {entry}")  # Plik
