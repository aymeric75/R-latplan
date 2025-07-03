import numpy as np




arr = np.array([
    [1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1],
])

print(arr)

arr = arr[~np.all(arr == 0, axis=1)]

print(arr)