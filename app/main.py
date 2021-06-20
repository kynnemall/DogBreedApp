import cv2
import streamlit as st
from PIL import Image

import models
# check this link to learn how to make predictions with streamlit:
# https://towardsdatascience.com/image-classification-of-uploaded-files-using-streamlits-killer-new-feature-7dd6aa35fe0

dog_detector_model = models.mobilenetv3_dog_detector()
breed_model = models.xception_model()

st.title("What dog breed are you?")
st.write("""
         Hi there!
         \nI'm a web app that has been trained to guess dog breeds!
         I can also tell if you're a human or a dog . . . 
         Either way, I'll guess what breed of dog you are!
         Just upload an image and let's get started!
         
         \nNo need to worry about your data; I have a poor memory so I won't be
         able to remember anything about you when you shut me down
         """, )

uploaded_file = st.file_uploader("Choose an image...", type=("jpg","png"))
if uploaded_file is not None:
    col1, col2 = st.beta_columns(2)
    
    # convert image to RGB instead of RGBA
    image = Image.open(uploaded_file).convert("RGB")
    col1.image(image, caption='Uploaded Image', use_column_width=True)
    col1.write("")
    col1.write("Classifying . . .")
    
    # predict human or dog
    certainty, res = models.detect_dog(image, dog_detector_model)
    col1.write(f"You are a . . . {res}! I'm {certainty:.1f}% sure about that")
    if res == "human":
        col1.write("Hmmm, but what breed would you be?")
    else:
        col1.write("Now, what breed are you?")

    # predict dog breed
    tensor = models.image_to_tensor(image)
    xception_features = models.extract_xception(tensor)
    print(xception_features.shape)
    results = breed_model.predict(xception_features)
    col1.write(results)