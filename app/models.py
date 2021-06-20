import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV3Small, MobileNet

def image_to_tensor(input_img):
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
    img = input_img.resize((224, 224))
    img_arr = np.array(img)
    tensor = np.expand_dims(img_arr, axis=0)
    return tensor

def mobilenet_breed_model():
    """
    Define architecture for custom MobileNet and load weights

    Returns
    -------
    model : CNN
        Custom MobileNet model with custom weights

    """
    mobile = MobileNet(input_shape=(224,224,3), classes=133,
                       include_top=False, weights=None)
    x = mobile.layers[-1].output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.1)(x)
    output = Dense(133, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=output)
    model.load_weights('weights/weights.mobilenet.best.hdf5')
    return model

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
    model.load_weights("weights/dogdetector.best.mobilenetv3-small.h5")
    return model

def detect_dog(input_img, model):
    """
    Use the model to predict whether the input image contains a dog or human

    Parameters
    ----------
    input_img : PIL Image
        uploaded user image
    model : CNN
        Custom MobileNetV3-Small model with custom weights

    Returns
    -------
    prediction : str
        Prediction of whether the input image contains a "Dog" or "Human"

    """
    img = input_img.resize((224, 224))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    result = "dog" if prediction > 0.5 else "human"
    if prediction > 0.5:
        certainty = (prediction - 0.5) / 0.5 * 100
    else:
        certainty = (0.5 - prediction) / 0.5 * 100
    return certainty, result
