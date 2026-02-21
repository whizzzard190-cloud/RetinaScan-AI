from models.model_loader import DRPredictor

predictor = DRPredictor()

# Replace with REAL retinal image path
IMAGE_PATH = "data/No_DR/002c21358ce6.png"

result = predictor.predict(IMAGE_PATH)

print(result)