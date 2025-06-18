
import numpy as np


a = [1, 2, 3, 3, 1, 3]
b = [1, 2, 3, 3, 1, 3]

a = np.array(a)
b = np.array(b)

a = tuple(np.ravel(a))
b = tuple(np.ravel(b))

assert a == b