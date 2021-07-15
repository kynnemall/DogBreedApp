import os
print(os.listdir())
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.xception import preprocess_input

import models
from lime import run_lime
# check this link to learn how to make predictions with streamlit:
# https://towardsdatascience.com/image-classification-of-uploaded-files-using-streamlits-killer-new-feature-7dd6aa35fe0

#dog_detector_model = models.mobilenetv3_dog_detector()
#breed_model = models.xception_model()
with open("dog_names.txt","r") as f:
    dog_names = [d.replace("_", " ").replace("\n", "") for d in f.readlines()]

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
    
    # convert image to RGB instead of RGBA
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying . . .")
    
    # predict human or dog
    certainty, res = models.detect_dog(image, dog_detector_model)
    if res == "human":
        st.write(f"You are a . . . {res}! I'm {certainty:.1f}% sure about that. Hmmm, but what breed would you be?")
    else:
        st.write(f"You are a . . . {res}! I'm {certainty:.1f}% sure about that. Now, what breed are you?")

    # predict dog breed
    tensor = models.image_to_tensor(image)
#    preprocessed_image = preprocess_input(tensor)
    results = breed_model.predict(tensor)
    breed = dog_names[np.argmax(results)]
    st.write(f"I think you would be a {breed}!")
    st.write("Soon I can explain why I think what breed you are, but you'll have to wait!")
