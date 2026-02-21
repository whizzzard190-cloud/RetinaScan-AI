from preprocessing.preprocess import preprocess_image

try:
    dummy = preprocess_image("cache/cache.pkl")
    print("Shape:", dummy.shape)
except Exception as e:
    print("Expected error:", e)