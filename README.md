## DogBreedApp
A simple web app for identifying dog breeds in colour images

### Libraries
* `streamlit` was used to build the web app
* `keras` was used to define model architectures and train models
* `openCV` and `scikit-learn` were used to train a logistic regression model on the results of Haar cascade classifiers to predict if human faces were present or not
* `scikit-image` was used to implement a local interpretable model-agnostic explanation (LIME) algorithm to better understand how the model makes predictions

### Datasets
* The dog breed dataset containing over 8000 images of 133 different breeds [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
* The human faces dataset (over 13000 images) can be found [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)
