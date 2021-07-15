import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from tensorflow.keras.applications.xception import preprocess_input

import models
from lime import run_lime
# check this link to learn how to make predictions with streamlit:
# https://towardsdatascience.com/image-classification-of-uploaded-files-using-streamlits-killer-new-feature-7dd6aa35fe0

st.set_page_config(
     page_title="Dog breed guesser"
)
track_tag = st.secrets["ANALYTICS_TAG"]

st.markdown(
    f"""
    <!-- Global site tag (gtag.js) - Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={track_tag}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
    
      gtag('config', '{track_tag}');
    </script>
    """, unsafe_allow_html=True
)

dog_detector_model = models.mobilenetv3_dog_detector()
breed_model = models.xception_model()
with open("app/dog_names.txt","r") as f:
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
    results = breed_model.predict(tensor)
    breed = dog_names[np.argmax(results)]
    if breed.lower().startswith(('a', 'e', 'i', 'o', 'u')):
        st.write(f"I think you would be an {breed}!")
    else:
        st.write(f"I think you would be a {breed}!")
    st.write("If you hang around for a minute or two, I'll tell you how I know that breed suits you!")
    top_superpixels = run_lime(np.array(image), tensor, breed_model)
    superpixels_img = Image.fromarray(top_superpixels)
    st.image(superpixels_img, caption="", use_column_width=True)
    st.write("So . . . ")
