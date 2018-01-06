from keras.models import load_model
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from keras import backend as Kbackend
import model_generator

model = load_model('model.h5')

# if os.path.isfile('sim_data/IMG/center_2016_12_01_13_36_54_429.jpg'):
	# print('got a file')
# else:
	# print('no file found')
img_right = cv2.imread('sim_data/IMG/center_2017_12_31_13_56_09_935.jpg')
img_left = cv2.imread('sim_data/IMG/center_2017_12_31_13_50_30_571.jpg')
img_straight = cv2.imread('sim_data/IMG/center_2017_12_31_13_51_17_541.jpg')

pred_right = model.predict(np.array(img_right[None,:,:,:]))
print('Right turn image: ', pred_right)
pred_left = model.predict(np.array(img_left[None,:,:,:]))
print('Left turn image: ', pred_left)
pred_straight = model.predict(np.array(img_straight[None,:,:,:]))
print('Straight image: ', pred_straight)



