import csv
import cv2
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

def generator(samples, batch_size=32, flip=True):
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			
			images = []
			measurements = []
			for batch_sample in batch_samples:
				# first column has the image file path
				source_path = batch_sample[0]
				filename = source_path.split("\\")[-1]
				current_path = 'sim_data\\IMG\\' + filename
				image = cv2.imread(current_path)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				#image = cv2.resize(image, (64,64))
				images.append(image)
				measurement = float(batch_sample[3])
				measurements.append(measurement)
				if flip:
					images.append(cv2.flip(image,1))
					measurements.append(-1*measurement)
			
			X_train = np.array(images)
			y_train = np.array(measurements)
			yield sklearn.utils.shuffle(X_train, y_train)

def return_nVidiaModel(lr=0.0001):
	from keras.models import Sequential
	from keras.layers import Flatten, Dense, Lambda, Activation, MaxPooling2D
	from keras.layers.convolutional import Convolution2D
	from keras.optimizers import Adam

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 -1.0,input_shape=(160,320,3)))

	model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

	model.add(Flatten())

	model.add(Dense(1164))
	model.add(Activation('relu'))
	model.add(Dense(100))
	model.add(Activation('relu'))
	model.add(Dense(50))
	model.add(Activation('relu'))
	model.add(Dense(10))
	model.add(Activation('relu'))
	model.add(Dense(1))


	model.compile(loss='mse', optimizer=Adam(lr=0.0001))
	
	return model

def return_csv_data(verbose=False):
	lines = []
	measurements = []
	with open('sim_data\driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
			measurement = float(line[3])
			measurements.append(measurement)
	if verbose:
		bins=np.linspace(-1.0, 1.0, 100)
		plt.hist(measurements,bins)
		plt.xlabel('Steering Angle')
		plt.ylabel('Frequency')
		plt.title('Histogram of Steering Angle Frequencies from Training Data')
		plt.show()
	return lines
	
	
def train_and_return_model(model=None, batch_size=32, flip=True, generate_weights=True, nb_epoch=5):


	train_samples, validation_samples = train_test_split(lines, test_size=0.2)
	
	train_generator = generator(train_samples, batch_size=batch_size, flip=True)
	validation_generator = generator(validation_samples, batch_size=batch_size, flip=True)
	
	model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*(1+flip), \
						validation_data=validation_generator, nb_val_samples=len(validation_samples), \
						nb_epoch=nb_epoch)
	return model

def save_model(model=None, name='model.h5'):
	model.save(name)

return_csv_data(verbose=True)
