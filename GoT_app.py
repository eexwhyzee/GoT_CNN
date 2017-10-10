import argparse
from GoT_utils import *
from GoT_model import *

# Parsing command line arguments 
arg_p = argparse.ArgumentParser()
arg_p.add_argument('-image_path', default='sample/')
arg_p.add_argument('-model_weights', default='saved_models/ver2.0_weights_final.hdf5')
args = vars(arg_p.parse_args())

IMAGE_PATH = args['image_path']
MODEL_WEIGHTS = args['model_weights']

sample = load_samples(IMAGE_PATH)

def GoT_algo(img_path):
	"""Function that takes a image(s) via file path(s) and makes predictions using the final CNN model.
	   Outputs a plot of image(s) with the predicted label.

	   Arguments:
	    img_path: File path of image(s) used to make a prediction.  
	"""
	
	fig = plt.figure()
	for i, image in enumerate(img_path):
		test_img = path_to_tensor(image).astype('float32')/255
		result = np.argmax(CNN(test_img, MODEL_WEIGHTS))
		img = cv2.imread(image)
		cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		ax = fig.add_subplot(4,5,i+1)

		if result == 1:
			plt.title('+')
		else:
			plt.title('-')
		plt.axis('off')
		plt.imshow(cv_rgb)
	plt.suptitle('Sample Images')
	plt.show()

GoT_algo(sample)