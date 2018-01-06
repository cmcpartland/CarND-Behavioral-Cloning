# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[modelarchitecture]: ./writeup_images/modelarchitecture.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


---
### Files Submitted

#### 1. My project includes the following files:
-	model.py – generates and saves the convolution neural network used to model the driver behavior
-	model_retrain.py – loads a previously saved model and trains it on new data
-	drive.py – connects to the driving simulator and uses the specified model to autonomously control the simulated vehicle
-	video.py – generates a video from the output frames from the simulator
-	model_trained.h5 – the trained model
-	output.mp4 – the video of the simulator in autonomous mode, being controlled by the model


#### 2. The Udacity simulator can be run using my mode with the following command:
```sh
python drive.py model_trained.h5
```

#### 3. 
Running model.py will attempt to train the model using a convnet on data generated using the Udacity simulator in training mode. The model is afterwards saved locally as model.h5. This module shows the pipeline used to pull and process the training data, and also shows the structure of the convnet used. 
Running model_retrain.py will attempt to load a previously saved model and train it on a new data set. This module is used to incrementally train the model in case more training data is needed to further train a somewhat successful model.


### Model Architecture

#### 1. Pre-Processing

The only pre-processing done is normalization on line 60 of model.py, which normalizes the pixel color values to a range of [-0.5, 0.5]. 

#### 2. Model Architecture

The model I used is a convolutional neural network based on nVidia’s end-to-end deep learning CNN for autonomous driving, found here https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
A visual model is provided by nVidia:
![alt text][modelarchitecture]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
