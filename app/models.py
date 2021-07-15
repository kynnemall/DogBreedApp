import numpy as np
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV3Small, Xception

def image_to_tensor(input_img, np_array=False):
    """
    Load image and prepare to be used as input to a CNN

    Parameters
    ----------
    input_img : Pillow Image object
        RGB image to be prepared

    Returns
    -------
    tensor : tensor
        4D array of shape (1, 224, 224, 3)

    """
    if np_array:
        func_image = Image.fromarray(input_img)
        print(func_image.size)
        img = np.resize(func_image, (224, 224, 3))
    else:
        print(input_img.size)
        img = np.resize(input_img, (224, 224, 3))
    tensor = np.expand_dims(img, axis=0)
    return tensor

def xception_model():
    """
    Build custom Xception model to predict dog breeds and load weights
    
    Returns
    -------
    Xception_model : CNN
        Custom Xception model to predict dog breeds
    """
    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    Xception_model.add(Dropout(0.2))
    Xception_model.add(Dense(133, activation='softmax'))
    Xception_model.load_weights("app/weights/weights.best.Xception.hdf5")
    model = Sequential()
    model.add(Xception(weights='imagenet', include_top=False))
    model.add(Xception_model)
    return model

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
    output = Dense(133, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=output)
    model.load_weights('app/weights/weights.mobilenet.best.hdf5')
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
    model.load_weights("app/weights/weights.best.dog-detector.mobilenetv3-small.h5")
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
