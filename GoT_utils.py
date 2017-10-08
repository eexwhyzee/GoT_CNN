from __future__ import absolute_import, division, print_function
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.preprocessing import image                  
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt    
import os                
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load the sample data and convert into tensors 
def load_samples(path):
	'''loads sample images without labels'''
	filenames = sorted(os.listdir(path))
	paths = ['%s%s' % (path,f) for f in sorted(filenames)]
	paths = np.array(paths)
	return paths

def load_dataset(path):
	data = load_files(path)
	GoT_files = np.array(data['filenames'])
	GoT_targets = np_utils.to_categorical(np.array(data['target']), 2)
	return GoT_files, GoT_targets

def path_to_tensor(img_path):
	'''loads RGB image as PIL.Image.Image type'''
	img = image.load_img(img_path, target_size=(224, 224))
	# convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
	x = image.img_to_array(img)
	# convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
	return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
	list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
	return np.vstack(list_of_tensors)

