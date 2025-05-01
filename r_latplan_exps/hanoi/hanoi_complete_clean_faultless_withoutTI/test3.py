import numpy as np

# Create an initial array of False
size = 10  # You can adjust this
result = np.full(size, False, dtype=bool)

# Loop and update result with OR operations
for i in range(5):  # Number of iterations
    temp = np.full(size, False, dtype=bool)
    
    # Set some random indices to True in temp array
    indices = np.random.choice(size, size=3, replace=False)  # Change size=3 as needed
    temp[indices] = True

    # OR operation to update result
    result = np.logical_or(result, temp)

    print(f"Iteration {i+1}, indices set to True: {indices}")
    print(f"Result so far: {result}\n")

# Final result
print("Final result:", result)
