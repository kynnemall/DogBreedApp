from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    tensor = np.expand_dims(img_arr, axis=0)
    return tensor

def dog_detector(img_path, resnet50_model=RESNET50_MODEL):
    img = resnet50.preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(ResNet50_model.predict(img))
    string_pred = ((prediction <= 268) & (prediction >= 151))
    return string_pred

def extract_Xception(tensor):
    model = Xception(weights='imagenet', include_top=False)
    result = model.predict(preprocess_input(tensor))
    return result

def load_Xception():
    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(input_shape=train_Xception.shape[1:]))
    Xception_model.add(Dropout(0.2))
    Xception_model.add(Dense(133, activation='softmax'))
    Xception_model.load_weights("weights/weights.best.Xception.hdf5")
    return Xception_model

def mobilenetv3_dog_detector():
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




