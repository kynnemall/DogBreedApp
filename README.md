## DogBreedApp
A simple web app for identifying dog breeds in colour images

### Libraries
* `streamlit` was used to build the web app
* `tensorflow.keras` was used to define model architectures and train models
* `scikit-image` was used to implement a local interpretable model-agnostic explanation (LIME) algorithm to better understand how the model makes predictions

### Datasets
* The dog breed dataset containing over 8000 images of 133 different breeds [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
* The human faces dataset (over 13000 images) can be found [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

### Model training
* Transfer learning was used to train models to predict whether an image contains a human or a dog and what dog breed is in the image
* An Xception model was trained to predict dog breeds. It scored 99% and 85% accuracy on the training and validation data, respectively
* MobileNetV3 was trained on equal numbers of dog and human images and achieved 100% accuracy on unseen test images
* MobileNetV3 is currently being trained to distinguish the dog breeds so as to reduce latency compared to the Xception model
