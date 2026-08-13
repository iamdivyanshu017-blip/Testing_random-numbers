import numpy as np
import os

def convert_input(binary_data):
    """
    Converts input into a numpy array of 0/1 integers.
    Accepts:
      - A string of '0'/'1' characters
      - A file path to a text file containing '0'/'1' characters
    Raises ValueError if the input isn't valid binary data.
    """
    # If it's a file path, read its contents first
    if isinstance(binary_data, str) and os.path.isfile(binary_data):
        with open(binary_data, 'r') as f:
            binary_data = f.read().strip()

    # Remove any whitespace/newlines just in case
    cleaned = "".join(c for c in binary_data if c in '01')

    if len(cleaned) == 0:
        raise ValueError("Input contains no valid binary (0/1) data.")

    if len(cleaned) < len(binary_data.replace('\n', '').replace(' ', '')):
        raise ValueError("Input contains characters other than 0 and 1.")

    return np.array([int(b) for b in cleaned], dtype=int)
