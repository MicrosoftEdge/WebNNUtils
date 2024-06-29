import numpy as np

# Create a 5D array with dimensions [1, 1, 4, 55, 64]
a = np.zeros((1, 4, 55, 64))
b = np.zeros((1, 4, 1, 64))
c = np.concatenate((a, b), axis=-2)


# You can also initialize it with specific values if needed:
# my_array = np.ones((1, 1, 4, 55, 64))  # Initialize with ones

print(f"Shape of the array: {c.shape}")
