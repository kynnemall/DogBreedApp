import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.xception import Xception, preprocess_input

def path_to_tensor(img_path):
    """
    Load image and prepare to be used as input to a CNN

    Parameters
    ----------
    img_path : str
        path to image

    Returns
    -------
    tensor : tensor
        4D array of shape (1, 224, 224, 3)

    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    tensor = np.expand_dims(img_arr, axis=0)
    return tensor

def extract_xception(tensor):
    """
    Extract features from the input tensor using the Xception model,
    pre-trained on imagenet, and return the output layer feature tensor

    Parameters
    ----------
    tensor : tensor
        4D array of shape (1, 224, 224, 3)

    Returns
    -------
    tensor : tensor
        features extracted by the Xception model

    """
    model = Xception(weights='imagenet', include_top=False)
    result = model.predict(preprocess_input(tensor))
    return result

def xception_model():
    """
    Build custom Xception model to predict dog breeds and load weights

    Returns
    -------
    Xception_model : CNN
        Custom Xception model to predict dog breeds

    """
    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(
        # shape is the same as result.shape[1:] from extract_Xception
        input_shape=train_Xception.shape[1:]
        ))
    Xception_model.add(Dropout(0.2))
    Xception_model.add(Dense(133, activation='softmax'))
    Xception_model.load_weights("weights/weights.best.Xception.hdf5")
    return Xception_model

def mobilenetv3_dog_detector():
    """
    Define architecture for custom MobileNetV3-Small and load weights

    Returns
    -------
    model : CNN
        Custom MobileNetV3-Small model with custom weights

    """
    pretrained_model = MobileNetV3Small(input_shape=(224,224,3),
                                    weights=None, include_top=False)
    # combine MobileNetV3 with extra layers for classification
    last_output = pretrained_model.layers[-1].output
    x = GlobalMaxPooling2D()(last_output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(pretrained_model.input, x)
    model.load_weights("weights/face.best.mobilenetv3-small.h5")
    return model

def detect_dog(img_path, model):
    """
    Use the model to predict whether the input image contains a dog or human

    Parameters
    ----------
    img_path : str
        path to image
    model : CNN
        Custom MobileNetV3-Small model with custom weights

    Returns
    -------
    prediction : str
        Prediction of whether the input image contains a "Dog" or "Human"

    """
    img = image.load_img(img_path, target_size=(224,224,3))
    prediction = model.predict(img)
    return prediction
