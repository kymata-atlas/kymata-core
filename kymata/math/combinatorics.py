from numpy import arange
from numpy.random import randint


def generate_derangement(n):  # approx 3ms runtime for n=400
    """
    Generates a derangement (permutation with no fixed point) of size `n`.
    """
    while True:
        v = arange(n)
        for j in range(n - 1, -1, -1):
            p = randint(0, j + 1)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            return v
