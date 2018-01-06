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

def generate_model():
	lines = []
	with open('sim_data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	lines.pop(0)
	training_set, validation_set = train_test_split(lines, test_size=0.2)
	batch_size = 64
	training_generator = generator(training_set, batch_size)
	validation_generator = generator(validation_set, batch_size)

	# source_path_center = lines[55][0]
	
	# filename_center = source_path_center.split("\\")[-1]

	# current_path_center = 'sim_data\\IMG\\' + filename_center
	
	# image_center = mpimg.imread(current_path_center)
	# print(float(lines[55][3]))
	# plt.imshow(image_center)
	# plt.show()

	from keras.models import Sequential
	from keras.layers import Flatten, Dense, Lambda, Activation
	from keras.layers.convolutional import Convolution2D, Cropping2D
	from keras.layers.pooling import MaxPooling2D
	
	# Separate function required with keras backend explicitly imported
	def resize(x):
		from keras import backend as K
		return K.tf.image.resize_images(x, (80,160))
	
	model = Sequential()
	# model.add(Lambda(resize, input_shape=(160,320,3)))
	model.add(Lambda(lambda x: (x/255.0)-.5, input_shape=(160,320,3)))
	model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Convolution2D(64,3,3,subsample=(1,1),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Convolution2D(64,3,3,subsample=(1,1),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='relu'))

	model.compile(loss='mse', optimizer='adam')
	model.fit_generator(training_generator, 
						samples_per_epoch = int(1000),
						validation_data = validation_generator,
						nb_val_samples=len(validation_set),
						nb_epoch = 1)

	model.save('model.h5')
	print('Model generated and saved.')


def generator(driving_log_lines, batch_size=32):
	# Create empty arrays to contain batch of features and labels
	num_samples = len(driving_log_lines)

	while True:
		shuffle(driving_log_lines)	
		for offset in range(0, num_samples, batch_size):
			batch_lines = driving_log_lines[offset:offset+batch_size]
			# images = np.empty((int(len(batch_lines)*3), 160, 320, 3), dtype=np.uint8)
			measurements = []
			images = []
			
			for i,line in enumerate(batch_lines):
				source_path_center = line[0]
				source_path_left = line[1]
				source_path_right = line[2]
				
				filename_center = source_path_center.split("\\")[-1]
				filename_left = source_path_left.split("\\")[-1]
				filename_right = source_path_right.split("\\")[-1]
				
				current_path_center = 'sim_data\\IMG\\' + filename_center
				current_path_left = 'sim_data\\IMG\\' + filename_left
				current_path_right = 'sim_data\\IMG\\' + filename_right
				
				image_center = mpimg.imread(current_path_center)
				image_left = mpimg.imread(current_path_left)
				image_right = mpimg.imread(current_path_right)
				# images[i, ...] = image_center
				# images[i+1, ...] = image_left
				# images[i+2, ...] = image_right
				images.append(image_center)
				images.append(image_left)
				images.append(image_right)
				
				steering_correction = 0.25
				steering_center = float(line[3])
				steering_left = steering_center + steering_correction
				steering_right = steering_center - steering_correction
				
				measurements.append(steering_center)
				measurements.append(steering_left)
				measurements.append(steering_right)
				
				
			# X_train = np.array(images)
			y_train = np.array(measurements)
			yield shuffle(np.array(images), y_train)
	
if (len(sys.argv) > 1):
	if sys.argv[1] == 'n':
		print('Genearting a new model...')
		generate_model()
		model = load_model('model.h5')
		print('Model model.h5 loaded successfully.')
else:
	try:
		model = load_model('model.h5')
		print('Model model.h5 loaded successfully.')
	except(OSError):
		print('Model model.h5 not found. Generating new model...')
		generate_model()
		model = load_model('model.h5')
		print('Model model.h5 loaded successfully.')

img_right = cv2.imread('sim_data/IMG/center_2017_12_31_13_56_09_935.jpg')
img_left = cv2.imread('sim_data/IMG/center_2017_12_31_13_50_30_571.jpg')
img_straight = cv2.imread('sim_data/IMG/center_2017_12_31_13_51_17_541.jpg')

pred_right = model.predict(np.array(img_right[None,:,:,:]))
print('Right turn prediction: ', pred_right, ', Actual: 0.08823529')
pred_left = model.predict(np.array(img_left[None,:,:,:]))
print('Left turn precition: ', pred_left, ', Actual: -0.03529412')
pred_straight = model.predict(np.array(img_straight[None,:,:,:]))
print('Straight image: ', pred_straight, ', Actual: 0')


