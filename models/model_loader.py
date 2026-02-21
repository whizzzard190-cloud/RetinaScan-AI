import numpy as np
import tensorflow as tf

from preprocessing.preprocess import preprocess_image
from preprocessing.labels import CLASS_NAMES
from utils.hash_utils import generate_image_hash
from utils.cache_utils import get_cached_result, store_result

MODEL_PATH = "models/dr_model.h5"


class DRPredictor:

    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)

    def predict(self, image_path):

        # Generate deterministic hash
        image_hash = generate_image_hash(image_path)

        # Check cache
        cached = get_cached_result(image_hash)
        if cached is not None:
            cached["cached"] = True
            return cached

        # Preprocess
        image = preprocess_image(image_path)

        # Inference
        preds = self.model.predict(image)[0]

        class_index = int(np.argmax(preds))
        confidence = float(preds[class_index] * 100)

        result = {
            "class": CLASS_NAMES[class_index],
            "confidence": round(confidence, 2),
            "cached": False
        }

        # Store in cache
        store_result(image_hash, result)

        return result