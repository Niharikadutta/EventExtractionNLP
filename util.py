import h5py
import json
import numpy as np

def load_embedding(path_name):
    with open(path_name+'.json', 'r', encoding='utf-8') as infile:
        dictionary = json.load(infile)
    with h5py.File(path_name+'.h5', 'r') as infile:
        matrix = np.zeros(infile['matrix'].shape, dtype=np.float32)
        infile['matrix'].read_direct(matrix)
    return dictionary, matrix
    
