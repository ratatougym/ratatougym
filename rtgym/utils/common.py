import hashlib


def hash_seed(seed, sensory_key):
    """Generate a hashed seed from base seed and sensory key.
    
    Ensures different sensory cells have different random number generators
    even when using the same base seed by incorporating the sensory key
    into the seed generation.
    
    Args:
        seed (int): Base seed value.
        sensory_key (str): Unique key for the sensory modality.
        
    Returns:
        int: Hashed seed value.
    """
    return seed + int(hashlib.sha256(sensory_key.encode('utf-8')).hexdigest(), 16) % (10 ** 8)