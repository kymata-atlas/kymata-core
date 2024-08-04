from numpy import arange
from numpy.random import randint


def generate_derangement(n, mod=int(1e9)):  # approx 3ms runtime for n=400
    assert n != 1, "An array of length one cannot be deranged"
    while True:
        v = arange(n)
        for j in range(n - 1, -1, -1):
            p = randint(0, j + 1)
            if v[p] % mod == j % mod or n == 1:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            return v
