## DogBreedApp
A simple web app for identifying dog breeds in colour images. Convolutional neural networks (CNNs) are capable of classifying images using learned feature representations for each class to be identified. State-of-the-art models were trained to determine if images contained a human or dog (MobileNetV3-Small) and then to predict which dog breed the image most resembles. The dog breeds were learned by training on a dataset containing 133 different breeds.

### Running this app
* Clone this repo
* Change directory to the `app` directory
* Install `tensorflow` and `streamlit` using pip
* Enter `streamlit run main.py` in a terminal. A browser tab containing the web app should open
* Upload a sample image and have fun!

### Libraries
* `streamlit` was used to build the web app
* `tensorflow` (version 2.5.0) was used to define model architectures and train models
* `scikit-image` and `scikit-learn` were used to implement a local interpretable model-agnostic explanation (LIME) algorithm to better understand how the model makes predictions. This algorithm was implemented according to a tutorial provided by Cristian Arteaga ([arteagac](https://arteagac.github.io/)), which can be found [here](https://nbviewer.jupyter.org/url/arteagac.github.io/blog/lime_image.ipynb)

### Datasets
* The dog breed dataset containing over 8000 images of 133 different breeds [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
* The human faces dataset (over 13000 images) can be found [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

### Model training
* Transfer learning was used to train models to predict whether an image contains a human or a dog and what dog breed is in the image
* An Xception model was trained to predict dog breeds. It scored 99% and 85% accuracy on the training and validation data, respectively
* MobileNetV3 was trained on equal numbers of dog and human images and achieved 100% accuracy on unseen test images
* MobileNetV3 is currently being trained to distinguish the dog breeds so as to reduce latency compared to the Xception model
