import numpy as np

def RandomSimplexVector(d=5, size=[1,] ):
    vec = np.random.exponential(scale=1.0, size=size + [d,])
    vec = vec / np.sum(vec, axis=-1).reshape(size + [1,])
    return vec