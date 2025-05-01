import numpy as np

arr = np.array([1, 0, 2, 0, 3, 0])
indices = np.where(arr == 0)[0]
print(indices)
