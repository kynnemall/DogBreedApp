import cv2
import keras
import numpy as np
import streamlit as st
from PIL import Image

from models import *
# check this link to learn how to make predictions with streamlit:
# https://towardsdatascience.com/image-classification-of-uploaded-files-using-streamlits-killer-new-feature-7dd6aa35fe0

st.title("Upload + Classification Example")

uploaded_file = st.file_uploader("Choose an image...", type=("jpg","png"))
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
