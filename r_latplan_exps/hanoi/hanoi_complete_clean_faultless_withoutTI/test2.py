import numpy as np

A = np.array([[1, 1, 0, 1],
              [1, 1, 0, 1]])

B = np.array([[1, 1, 0, 0],
              [0, 0, 0, 1]])

# print(A - B)


entails = np.all((A & B) == B, axis=1)

print(entails)


# A[entails] = A[entails] - B[entails]

# print(A)