import pickle
import os

CACHE_PATH = "cache/cache.pkl"


def load_cache():
    if not os.path.exists(CACHE_PATH) or os.path.getsize(CACHE_PATH) == 0:
        return {}

    try:
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    except:
        return {}


def save_cache(cache_dict):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache_dict, f)


def get_cached_result(image_hash):
    cache = load_cache()
    return cache.get(image_hash, None)


def store_result(image_hash, result):
    cache = load_cache()
    cache[image_hash] = result
    save_cache(cache)