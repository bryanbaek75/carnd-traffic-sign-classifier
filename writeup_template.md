#**Traffic Sign Recognition** 

##Write Up of project No.2, Term1 

###Bryan Baek

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/data_web1.jpg "Traffic Sign 1"
[image5]: ./examples/data_web2.jpg "Traffic Sign 2"
[image6]: ./examples/data_web3.jpg "Traffic Sign 3"
[image7]: ./examples/data_web4.jpg "Traffic Sign 4"
[image8]: ./examples/data_web5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32,32,3).
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because it will clarify the recognizable characteristic of the traffic signs data better then colored image.  Also, it will be more robust with small set of training data. Traffic sign itself is very recognizable due to the high contrast ratio, so it must works well with grayscale. 

Here is an example of a traffic sign image after gray scaling.

![alt text][image2]

As a last step, I normalized the image data because the normalized data works better with initial network parameter. most hyper parameter is -1 to 1 scale, so normalizing image data from RGB to -1 to 1 scale will work better with initial hyper parameters.  

I tried not to generate additional data to see if the model change will work better even with biased training data distribution.  

My final model consisted of the following layers:

|          Layer           |               Description                |
| :----------------------: | :--------------------------------------: |
|          Input           |         32x32x1 Grayscale image          |
|     Convolution 3x3      | 1x1 stride, valid padding, outputs 28x28x32 |
|           RELU           |                                          |
|       Max pooling        |      2x2 stride,  outputs 14x14x32       |
|     Convolution 3x3      | 1x1 stride, valid padding, outputs 10x10x32 |
|           RELU           |                                          |
|       Max pooling        |       2x2 stride,  outputs 5x5x32        |
|         Flatten          |                                          |
|     Fully connected      |          input 800, output 240           |
|           RELU           |                                          |
|     Fully Connected      |          input 240, output 120           |
|           RELU           |                                          |
| Drop Out (training only) |           keep probability 0.5           |
|     Fully Connected      |           input 120, output 43           |
|         Softmax          |                   etc.                   |



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an adam optimizer. softmax cross entropy. Others are following. 

- batch size = 128
- epochs = 100
- learning rate = 0.001 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* average training set accuracy of 99.4% ~ 100%  (After several epoch)
* validation set accuracy of  73.1% (epoch1) to 96.5% , it is varied along the epoch.  
* test set accuracy of 94.1%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?  --> The original architecture the proven LeNet basic Model. It is simple and match well with given input data. 
* What were some problems with the initial architecture? --> Accuracy rate is stocked at 89%. it was not improved. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting. --> Applying drop out improved much better. Modification of layer solved the problem. 
* Which parameters were tuned? How were they adjusted and why? --> The data was biased, so the overfitting was suspected. More richer fully connected layer and more depth on convolution layer might solve the problem, I expected. Deeper network may improve the data distribution problem cause it is not easily biased with small data set. And LeNet originally designed for MNIST data set. MNIST data set final output is only 10, compared to that, problem space was 43. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? --> Drop out prevent overfitting well to some level and more depth on convolution and more fully connected layer will solve overfitting better, I expected. 

If a well known architecture was chosen:
* What architecture was chosen? --> LeNet 
* Why did you believe it would be relevant to the traffic sign application? --> As I described above.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? --> it was 89% initially, and improved 93.9 ~ 94.5% ( Test set ) later. 


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|       Image       |          Prediction          |
| :---------------: | :--------------------------: |
|    Bumpy Road     | Dangerous curve to the right |
|       Stop        |             Stop             |
|   Slippery Road   |        Slippery Road         |
| Children Crossing |      Children Crossing       |
|     Road Work     |          Road Work           |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of German traffic signs data, it is just good.  And after resizing to 32*32 pixels, web data image was not easy to recognize even for human. 64 by 64 resolution might improve the accuracy better.   

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Keep right sign (probability of 0.95), and the image does contain a Bumpy road sign. The top five soft max probabilities were

| Probability |      Prediction      |
| :---------: | :------------------: |
|     .95     |      Keep right      |
|     .05     |      Bumpy road      |
|     .00     |      Road work       |
|     .00     |     Double curve     |
|     .00     | Go straight or right |

For the second image, the model is sure that this is a Stop sign (probability of 1.00), and the image does contain a Stop sign. The top five soft max probabilities were

| Probability |      Prediction      |
| :---------: | :------------------: |
|    1.00     |         Stop         |
|     .00     | Go straight or left  |
|     .00     | Speed limit (30km/h) |
|     .00     |      Keep right      |
|     .00     |   Turn left ahead    |

For the third image, the model is sure that this is a Slippery road (probability of 1.00), and the image does contain a Slippery road sign. The top five soft max probabilities were

| Probability |                Prediction                |
| :---------: | :--------------------------------------: |
|    1.00     |              Slippery road               |
|     .00     |       Dangerous curve to the left        |
|     .00     |          Wild animals crossing           |
|     .00     | No passing for vehicles over 3.5 metric tons |
|     .00     |                Keep right                |

For the fourth image, the model is sure that this is a Children crossing (probability of 1.00), and the image does contain a Children crossing. The top five soft max probabilities were

| Probability |          Prediction          |
| :---------: | :--------------------------: |
|    1.00     |      Children crossing       |
|     .00     | Dangerous curve to the right |
|     .00     |            Yield             |
|     .00     |     Speed limit (20km/h)     |
|     .00     |     Speed limit (30km/h)     |

For the fifth image, the model is sure that this is a Road work (probability of 1.00), and the image does contain a Road work sign. The top five soft max probabilities were

|      Probability      |          Prediction          |
| :-------------------: | :--------------------------: |
| 1.00 (1.00000000e+00) |          Road work           |
| .00 (2.40010550e-12)  |  Road narrows on the right   |
| .00 (3.20807427e-14)  | Dangerous curve to the right |
| .00 (2.72722159e-17)  |    Wild animals crossing     |
| .00 (1.28470025e-17)  |      Bicycles crossing       |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


