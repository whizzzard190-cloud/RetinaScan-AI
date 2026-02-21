from utils.hash_utils import generate_image_hash
from utils.cache_utils import store_result, get_cached_result

fake_path = "cache/cache.pkl"

h = generate_image_hash(fake_path)

store_result(h, {"class": "Test", "confidence": 99})

result = get_cached_result(h)

print("Hash:", h)
print("Cached:", result)