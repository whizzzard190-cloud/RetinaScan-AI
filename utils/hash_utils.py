import hashlib


def generate_image_hash(image_path):
    """
    SHA256 hash for deterministic medical inference
    """

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    return hashlib.sha256(image_bytes).hexdigest()