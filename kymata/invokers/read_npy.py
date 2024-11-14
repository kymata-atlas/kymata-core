import numpy as np
import sys

# The file path and index are passed as arguments
file_path = sys.argv[1]
index = int(sys.argv[2])

# Load the .npy file
array = np.load(file_path)

# Extract the specific element at array[index]
element = array[index]

# Ensure the element is an array or list (if needed)
# If it's a single value, convert it to a list with one element for uniformity
if isinstance(element, np.ndarray) or isinstance(element, list):
    # Print values space-separated for easier parsing in shell
    print(" ".join(map(str, element)))
else:
    print(element)
