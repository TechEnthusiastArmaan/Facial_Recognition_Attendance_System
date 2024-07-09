import pickle
import numpy as np

# Load and inspect the faces_data.pkl file
with open('attendence system\\data\\faces_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
print(type(data))
if isinstance(data, tuple):
    print(f"Tuple length: {len(data)}")
    for i, item in enumerate(data):
        print(f"Element {i} type: {type(item)}")
        if isinstance(item, list) or isinstance(item, np.ndarray):
            print(f"Element {i} length: {len(item)}")
