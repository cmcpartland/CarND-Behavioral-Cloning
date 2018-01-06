import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as Kbackend
from keras.models import load_model
from sklearn.model_selection import train_test_split
import sys
from sklearn.utils import shuffle
import matplotlib.image as mpimg

# Define a generator which uses the new data folder
def generator(samples, batch_size=32):
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
				current_path = 'data2\\IMG\\' + filename
				image = cv2.imread(current_path)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				#image = cv2.resize(image, (64,64))
				images.append(image)
				images.append(cv2.flip(image,1))
				measurement = float(batch_sample[3])
				measurements.append(measurement)
				measurements.append(-1*measurement)
			

			X_train = np.array(images)
			y_train = np.array(measurements)
			yield shuffle(X_train, y_train)

# Attempt to load the previously saved model
try:
	model = load_model('model.h5')
	print('Model model.h5 loaded successfully.')
except(OSError):
	print('Model model.h5 not found.')
	model = None

# Read each line from the training data CSV file
lines = []
with open('data2\driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
lines.pop(0)

# Create a training and validation set using train_test_split from sklearn
training_set, validation_set = train_test_split(lines, test_size=0.2)
batch_size = 64
training_generator = generator(training_set, batch_size)
validation_generator = generator(validation_set, batch_size)

from keras.optimizers import Adam

# Compile the model and fit the new data to it
model.compile(loss='mse', optimizer = Adam(lr=0.001))
model.fit_generator(training_generator, 
					samples_per_epoch = len(training_set)*2,
					validation_data = validation_generator,
					nb_val_samples=len(validation_set),
					nb_epoch = 5)

# Save the model
model.save('model_trained.h5')
print('New model saved --> model_trained.h5')