# **Traffic Sign Recognition** 

## A Simple CNN Architecture inspired from Le-Net 5 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./assets/hist.png "Histogram"
[image2]: ./assets/0.png "0"
[image3]: ./assets/1.png "1"
[image4]: ./assets/2.png "2"
[image5]: ./assets/3.png "3"
[image6]: ./assets/4.png "4"
[image7]: ./assets/5.png "5"
[image8]: ./assets/gray.png "Gray"
[image9]: ./web/20speed.jpg "20speed"
[image10]: ./web/animalscrossing.jpg "Animals Crossing"
[image11]: ./web/keepright.jpg "Keep Right"
[image12]: ./web/nopassing.jpg "No Passing"
[image13]: ./web/roadwork.jpg "Road Work"
[image14]: ./web/stop.jpg "Stop"
[image15]: ./web/yield.jpg "Yield"


---

### Data Set Summary & Exploration

#### 1. Summary of Data

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of Dataset

A visualization of the frequency of samples per class in the Training dataset can be seen below.

![alt text][image1]

Some samples from the dataset can be seen below.

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

### Design and Test a Model Architecture

#### 1.Data Preprocessing

As a first step, I decided to convert the images to grayscale because it reeduces the number of channels which reduces the training time. 

Here is an example of a traffic sign image after grayscaling.

![alt text][image8]

I also normalised the pixel values by subtracting 125 and dividing by 125. This restraints them to be in the range of -1 to +1. This makes sure that while backpropogation, the gradients don't blow up.


#### 2. Final Model

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x12 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 8x8x24 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 4x4x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x24 					|
| Flatten				| 												|
| Dense					| outputs 240 									|
| RELU					|												|
| Dropout				|Keep_prob 0.7									|
| Dense					| outputs 240 									|
| RELU					|												|
| Dropout				|Keep_prob 0.7									|
| Dense					| outputs 120 									|
| RELU					|												|
| Dropout				|Keep_prob 0.7									|
| Dense					| outputs 43 									|


#### 3. Training the model

To train the model, I used the Adam Optimizer with a learning rate of 0.001 which I ran for a total of 25 epochs with a batch size of 128. As this was a multi-class classification, I used the Cross Entropy function as my loss function.

#### 4. Analysis of the model

My final model results were:
* training set accuracy of 99.0%
* validation set accuracy of 95.1% 
* test set accuracy of 92.3% over

Initially I tried the vanilla LeNet-5 but even for a small data, the model was not overfitting which indicated that a more complex network was required. After increasing the number of Convolution and Dense layers, the training accuracy was almost hitting 99% which indicated Overfitting. Hence I used Dropout layers after all the Dense layers with a dropout probability of 0.3. This increased the validation accuracy drastically.


### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image9] ![alt text][image12] ![alt text][image13] ![alt text][image15] 
![alt text][image10] ![alt text][image11] ![alt text][image14] 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (20km/h)	| Speed Limit (20km/h)							| 
| Wild Animals Crossing	| Wild Animals Crossing							|
| Keep Right			| Keep Right									|
| No Passing      		| Children Crossing				 				|
| Road Work 			| Road Work          							|
| Stop      			| Stop         									|
| Yield      			| Yield        									|


The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.6%. These images are taken randomly from the internet.

For the fourth image, the model is relatively sure that this is a End of No Passing sign (probability of 0.6), and the image does not contain a stop sign but No Passing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .843         			| Children Crossing								| 
| .107     				| Right-of-way at the next intersection			|
| .003					| Vehicles over 3.5 metric tons prohibited		|
| .0005	      			| Slippery road					 				|
| .0004				    | No Passing         							|

