from keras.layers import Conv2D, Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization 


def CNN(x, weights):
	"""CNN implementation used for making predictions.

		Arguments: 
		 x: Input image with shape (224,224,3).
		 weights: File path directing to saved weights.

	    Returns: 
	     Prediction for input image.
	"""

	model = Sequential()

	model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', input_shape=(224,224,3)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(filters=64, kernel_size=3, strides=2, padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))


	model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same')) 
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(filters=128, kernel_size=3, strides=2, padding='same')) 
	model.add(BatchNormalization())
	model.add(Activation('relu'))


	model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same')) 
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(filters=256, kernel_size=3, strides=2, padding='same')) 
	model.add(BatchNormalization())
	model.add(Activation('relu'))


	model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same')) 
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(filters=512, kernel_size=3, strides=2, padding='same')) 
	model.add(BatchNormalization())
	model.add(Activation('relu'))


	model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same')) 
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(filters=512, kernel_size=3, strides=2, padding='same')) 
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv2D(filters=1024, kernel_size=3, strides=2, padding='same')) 
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(1000))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.7)) 

	model.add(Dense(1000))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5)) 

	model.add(Dense(2, activation='softmax'))

	model.load_weights(weights)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	
	return model.predict(x)