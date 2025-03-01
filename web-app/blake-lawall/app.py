import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 128

# def predict_pneumonia(image):
def predict_pneumonia(image):

    # Prediction function
    model = tf.keras.models.load_model('improved_pneumonia_model.keras')
    # Preprocess image
    image = np.expand_dims(image, axis=-1)
    resize = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    resize = resize / 255.0
    
    # Make prediction
    prediction = model.predict(np.expand_dims(resize, 0))
    yhat = model.predict(np.expand_dims(resize, 0))
    pred_value = yhat[0][0]

    # Return prediction
    return {
        'NORMAL': 1 - float(pred_value),
        'PNEUMONIA': float(pred_value)
    }

# Build the Gradio interface 
gr_interface = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(image_mode='L'),
    outputs=gr.Label(num_top_classes=2),
    title='Pneumonia X-Ray Classifier',
    description='Upload a chest X-ray image to check for pneumonia.',
    examples=[
        # Optional: Add example images here
        ["val/NORMAL/NORMAL2-IM-1427-0001.jpeg"],
        ["val/NORMAL/NORMAL2-IM-1430-0001.jpeg"],
        ["val/NORMAL/NORMAL2-IM-1431-0001.jpeg"],
        ["val/NORMAL/NORMAL2-IM-1436-0001.jpeg"],
        ["val/NORMAL/NORMAL2-IM-1437-0001.jpeg"],
        ["val/NORMAL/NORMAL2-IM-1438-0001.jpeg"],
        ["val/NORMAL/NORMAL2-IM-1440-0001.jpeg"],
        ["val/NORMAL/NORMAL2-IM-1442-0001.jpeg"],
        ["val/PNEUMONIA/person1946_bacteria_4874.jpeg"],
        ["val/PNEUMONIA/person1946_bacteria_4875.jpeg"],
        ["val/PNEUMONIA/person1947_bacteria_4876.jpeg"],
        ["val/PNEUMONIA/person1949_bacteria_4880.jpeg"],
        ["val/PNEUMONIA/person1950_bacteria_4881.jpeg"],
        ["val/PNEUMONIA/person1951_bacteria_4882.jpeg"],
        ["val/PNEUMONIA/person1952_bacteria_4883.jpeg"],
        ["val/PNEUMONIA/person1954_bacteria_4886.jpeg"]
    ]
)

# Launch the app
if __name__=="__main__":
    gr_interface.launch()
