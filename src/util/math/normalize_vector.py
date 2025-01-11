import numpy as np


def normalize_vector(vector):
    """Normalize a vector.

    Normalizes a vector to length one.

    Args:
                    vector: Numpy array of the vector.

    returns:
          Normalized numpy array of the vector.

    Raises:
                    N/A
    """
    dim = vector.shape[0]
    vector_components = []
    for d in range(dim):
        vector_components.append(vector[d])

    norm = np.sqrt(sum([np.power(v, 2) for v in vector_components]))

    vector_norm = []
    for d in range(dim):
        vector_norm.append(-vector[d] / norm)

    return vector_norm


def normalize_complex_vector(vector):
    """Normalize a complex vector.

    Normalizes a complex vector to length one.

    Args:
                    vector: Numpy array of the complex vector.

    returns:
          Normalized numpy array of the complex vector.

    Raises:
                    N/A
    """
    vector_components = []
    vector_components.append(np.real(vector))
    vector_components.append(np.imag(vector))

    norm = np.sqrt(sum([np.power(v, 2) for v in vector_components]))

    vector_norm = []
    vector_norm.append(-np.real(vector) / norm)
    vector_norm.append(-np.imag(vector) / norm)

    return np.array(vector_norm[0] + vector_norm[1] * 1j)
