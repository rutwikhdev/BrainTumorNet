import h5py as h5 
import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

np.random.seed(1)

DATADIR = os.getcwd() + '/brain_tumor_dataset'
subdirs = [os.path.join(DATADIR, x) for x in os.listdir(DATADIR)] # train and validation paths

def create_dataset(data, labels, filename):
    """Creates hdf5 file with two datasets(data, labels)"""
    hFile = h5.File(filename, 'w')
    hFile.create_dataset('dataset', data=data)
    hFile.create_dataset('labels', data=labels)
    hFile.close()

def load_dataset(filename):
    """Retrives the dataset and labels"""
    hFile = h5.File('./datasets/' + filename, 'r')
    data = hFile['dataset'][:]
    labels = hFile['labels'][:]
    hFile.close()
    return data, labels

def readFile2(subdirs): # train and validation paths
    lst = []
    labels = []
    breedpaths = [os.path.join(subdirs, x) for x in os.listdir(subdirs)]
    for i in range(len(breedpaths)):
        for img in os.listdir(breedpaths[i]):
            img = Image.open(os.path.join(breedpaths[i], img)).convert('L')
            print(len(img.size))
            img = img.resize((240,240))
            lst.append(np.array(img))
            labels.append([i])
    d, l = unison_shuffled_copies(np.array(lst, dtype=np.uint8), np.array(labels, dtype=np.uint8).reshape(len(labels), 1))
    return d, l
    
    
"""compressing the image dataset in hdf5 format"""
# data, labels = readFile2(subdirs[0])
# create_dataset(data, labels, 'test.h5')
# print(data.shape)
# print('Dataset created!')

# retrieved = load_dataset()
# data = np.array(retrieved)
# print(data[0])
# print('worked')

#print(DATADIR)
"""Shape of the nparrays going in h5 should be uniform"""

