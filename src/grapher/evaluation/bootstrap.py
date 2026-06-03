import random

def bootstrap_sample(items, k: int, seed: int):
    rng = random.Random(seed); return [rng.choice(items) for _ in range(k)]
