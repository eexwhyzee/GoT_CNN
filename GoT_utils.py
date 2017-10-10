from __future__ import absolute_import, division, print_function
from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix, classification_report
import itertools
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


def load_samples(path):
	"""Loads sample images without labels.

	   Arguments: 
	   	path: Directory path of images.

	   Returns: 
	    A list of individual paths for each image file.
	"""

	filenames = sorted(os.listdir(path))
	paths = ['%s%s' % (path,f) for f in sorted(filenames)]
	paths = np.array(paths)
	return paths

def load_dataset(path):
	"""Loads dataset with labels (directory for each label type).

	   Arguments: 
	    path: Directory path of images.

	   Returns: 
	    GoT_files: List of indiviual paths for each image file.
	   	GoT_targets: List of target labels for each image file. 
	"""

	data = load_files(path)
	GoT_files = np.array(data['filenames'])
	GoT_targets = np_utils.to_categorical(np.array(data['target']), 2)
	return GoT_files, GoT_targets

def path_to_tensor(img_path):
	"""Loads RGB image as PIL.Image.Image type and convert to tensor.

	   Arguments:
	    img_path: Path of RGB image file.

	   Returns:
	    4D tensor with shape (1,224,224,3).
	"""

	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
	"""Stacks individual image tensors together.

	   Arguments:
	    img_paths: List of image file paths.

	   Returns:
	    Stack of tensors, each tensor with shape (224,224,3).
	"""

	list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
	return np.vstack(list_of_tensors)

def cm_plot(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
	"""Plots confusion matrix.

	   Arguments:
	    cm: Confusion matrix created using sklearn.
	    classes: List of class names (labels).
	    cmap: Color map scheme for plot.
	"""

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' 
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')