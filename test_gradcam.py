import tensorflow as tf
from preprocessing.preprocess import preprocess_image
from gradcam.gradcam_utils import make_gradcam_heatmap, save_gradcam

IMAGE_PATH = "data/No_DR/002c21358ce6.png"

model = tf.keras.models.load_model("models/dr_model.h5")

img = preprocess_image(IMAGE_PATH)

heatmap = make_gradcam_heatmap(img, model)

path = save_gradcam(IMAGE_PATH, heatmap)

print("GradCAM saved to:", path)