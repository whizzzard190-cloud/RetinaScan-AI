import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 320

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Invalid image path")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32)

    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    return image