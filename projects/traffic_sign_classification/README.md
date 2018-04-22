**Traffic Sign Recognition** 

The goal of this project is to design a neural network that can identify and classify traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)

The steps taken to implement this & the results achieves are described below in this document.

---
#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/stoicio/traffic_sign_classification)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Using numpy, I printed out the shape of all the available dataset
* The size of training set is - __*34799*__
* The size of the validation set is  __*4410*__
* The size of test set is __*12630*__
* The shape of a traffic sign image is __*32 x 32 x 3*__
* The number of unique classes/labels in the data set is __*43*__

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the available classes. 

![alt text][image1]

As we can see here, the distribution of the training data is very skewed, with some classes having thousands of images, while some classes only have a few hundreds. This will lead to overfitting for classes with large number of image data. Lets fix this by augumenting and generating some fake data so that the training data is well distributed across classes.

I also tried to visualize some images in the dataset by randomly choosing 3 images out of each class and plotting them

![alt text][image1_1]

Running the visualization a few times and looking at the test data, there is a lot of variation for the following factors
* Illumination - Some images are well lit, while some images are almost dark
* Scale of the sign

While CNNs are resistant to slight illumination variations, we can accelerate training a lot by normalizing the images for brightnes. Scale variance in the test data is a good characteristic. This makes the model robust to images captured at different distances.
There is also no variance for the following factors :
* Rotation
* Sheer (affine transformation)
* Placement of the sign with respect to image centre

Almost all the images I observed were upright & sign was parallel to the image plane. This might not be true in real world conditions. To make the model robust to real world conditions, I will try and augument the given images by introducing rotation & random affine transformations

Further, I plotted out each channel of the image separately to deduce if the images were stored as a RGB / BGR image.
From the below images, I was able to tell that the image channel order was Red, Green and Blue
![alt text][image1_2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

At first, I tried to attemp the preprocessing steps described in the paper [linked](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) 
* Conversion to YUV color space and normalizing only the Y channel, effectively equalizing the brighness of the images, while preserving the color features in the U & V channels made immediate sense intuitively. The motivation behind this was the variance in the illumination found in the training set as mentioned above.
* However, i wasn't able to implement the global and local normalization by myself. I did try the steps mentioned [at this link](http://bigwww.epfl.ch/demo/jlocalnormalization/) using some OpenCV methods, but the results were not satisfactory. I presume, i wasn't able to land the right parameters for the kernels. I would like to come back to this.

**Adaptive Histogram Equalization :**
* I resorted to a fallback attempt of using adaptive histogram equalization , which effectively transforms the image to have uniform illumination / brightness. However this also meant that I lost the color features in the image because of conversion to grayscale.
* But I believe this is a better way to normalize as it considers local features, instead of global normalization techniques that was introduced in the lectures & labs (which pushes for zero mean across the entire image)
* the following image shows some of the training data after it has been preprocessed

![alt text][image2]

In the exploratory step, I had mentioned some invariance in the data for the following factors:
* Rotation - In all the training data observed, the traffic sign was upright
* Affine transforms - All the training data observed was parallel to camera plane
* Translations - All the training data observed was in almost centred in the image frame

Further, more data was required to uniformly distribute the training data over all the classes to prevent overfitting

Augumentation Steps (Used values from the Sermanet paper):
* Oriiginal image rotated by +/- 15 degrees
* Shift images by +/-2 pixels horizontally & vertically 
* Scale images to 0.9 to 1.1 of their original size
* Sheer images (affine transformation) 
* NO horizontal / vertical flipping as that would change the labels of some of the images ( and would have taken a lot of painstaking effort to handpick those that could have been flipped)
* With the above transformations, additional images were generate for each class so that all the classes had close to 2500 images . Distribution of images after augumentation can be seen below 

![alt text][images_2500]

#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1  Brightness equalized grayscale image	| 
| Convolution 5x5   	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	    	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5 (L1)  | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	  (L2)	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten L1 (L3)		| Output 10 x 10 x 16       					|
| Flatten L2 (L4)		| Output 5 x 5 x 16       						|
| Linear Classifier		|	Concat L3 & L4 - Output 2000 features    	|
| Dropout 50% 			|	Drop 50% of the activations for training	|
| Fully Connected		|	2000 features to get 43 classifications 	|


#### 3. Training Parameters

To train the model, I used the Sermanet architecture with the following parameters
* Adam Optimizer - because it takes care of decaying the learning rate with epochs & suggested default optimizer by many including Karpathy
* Batch Size of 256
* 30 Epochs
* Learning rate 0.001
* Dropout for fully connected layer : 0.5
* Dropout for convolution layers : 0.25

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* training set accuracy of __*98.8%*__
* validation set accuracy of __*96.8 %*__ 
* test set accuracy of __*95.5 %*__

__*Training & Validation accuracy across epochs*__

![alt text][training_epochs] ![alt text][validation_epochs]

I tried both LeNet (without dropouts) and Sermanet architecture (from the paper linked). 

While I was able to quickly get past 90% using LeNet I hit a plateau and couldn't get validation accuracy past 93 - 94 %.  Also the training accuracy very quickly crossed 99%. The difference between the training and validation graphs suggested that there was some overfitting in the model. 

Using Sermanet I was able to achieve greater than 95% validation accuracy but the problem with overfitting persisted.
After some trial and error with the hyperparameters, I went back to the data augmentation step to generate more image data. Previously I had only generate enough fake data so that the number of images in each class was close to average images in each class. I realized that was not enough to get an uniform distribution across classes and  generated more fake data so that each class has close to 2500 images. The distribution of the augumented data can be seen below

![alt text][images_mean] ![alt text][images_2500]

I also decided to add dropouts as suggested in the lectures to prevent overfitting. I used a dropout of 50% in the fully connected layer & dropout of 25% in the convolution layers.
The motivation behind using dropouts in the convolution layers was, that the intuitive thought that this would make the model more robust to occlusions ( Dropping activations in the convolution layer, where the local features like edges, corners & line segments were detected - is a way to simulate occlusions ? Retrospectively, I think a similar effect could have been achieved if I had randomly blacked out a 5x5 ROI in the test data )

After these modifications, training the model resulted in better Training and Validation accuracy. The training and validation accuracy also tracked close to each other with close to 2% difference. 

Further tuning resulted in diminishing returns.  i am also curious to see if there are better ways to tuning the hyper parameters ( or an automated way ? where the training process involves giving a range of hyper parameters and the model that gets the best validation accuracy is automatically choosen)

PS: As I have mentioned below, after visualizing the CNN output, it would be interesting to see what accuracy gains can be obtained by doing the following two things
* Try the YUV color space - to make the model more robust to colored features (Eg: General Caution vs Traffic Signals)
* Add more filters in the convolution step - to check if the CNN detects more unique features / local features.

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image5_1] ![alt text][image5_2]![alt text][image5_3] ![alt text][image5_4] ![alt text][image5_5]

The above images might be difficult to classify because,
* First image is not cropped to include only the traffic sign and has a lot of other features arounds it. in the Training data, almost all the image are well cropped to include only the traffic sing
* In the second image "Go straight or left" the illumination across the traffic sign changes, 
* in the third image, "30 Km/hr speed limit"  part of the sign in obscured by reflection and the red circle is not clearly visible
* In the fourth image, "General Caution" the sign is rotated and also a skewed a bit. 
* In the fifth image, "Speed limit 70 km/hr" the  image's brightness is too low.

Here are the results of the prediction:

| Image			                       |     Prediction	        | Correct Prediction  |
|:------------------------------------:|:----------------------:|:-------------------:| 
| End of all speed and passing limits  | No Vehicles   			| No				  | 
| Go Straight or Left    			   | Go Straight or Left 	| Yes				  |
| Speed limit (70km/h)				   | Speed limit (70km/h)	| Yes				  |
| General Caution      		           | General Caution		| Yes 		 		  |
| Speed limit (70km/h)			       | Speed limit (70km/h)   | Yes  				  |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. While this is less than the accuracy I saw from test data set, I would like to experiment with more new images when I obtain them from the web.
( The course & GTSRB is so popular its hard to weed through all the search results and eliminate images that are not part of the dataset itself ! )

#### 2. Softmax probabilities

* For the first image, which was a wrong prediction, the correct class is not in the Top 5. 
    *   Even for the predictions made, the model is not sure. It appears that the circle was picked as the most prominent feature and the next prominent feature is th 45 degree line which is evident from the top 2 predictions from the model. 
    *   This could be due to the fact that the model is UNDERFITTED for images that are not cropped close enough to have only the traffic sign and include other background features. Ideally, i would imagine, before this model is used for inference, other localization techniques should be used to get a tight bounding box around the sign and the images preprocssed to contain only the bounding box.
* For Image 4, General Caution, though the model is very sure ( > 95% ) the next next highest prediction is Traffic Signals which is visually very similar to General Caution sign. 
    *   this could be an __artifact of converting the images to gray scale__. With the color features the model would have been able to distinguish between the black vertical line and the three colored circles in the __Traffic Signals__ sign (which is close enough to resemble a single line)
* For Images, 2, 3, 5 the model is almost 100% certain of its prediction. For Image 3, the 50 km/h sign is the second highest probable prediction although the certainity for that is very low

![alt text][softmax]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Vizualizing the CNN activation was quite fascinating :) Its interesting that he CNN has picked up interesting features in the image without any kind of supervision ! 

* __*FeatureMap 2*__ appears to have picked up the brightest parts of the traffic signs, which would help it distinguish between the the traffic sign background & the symbol as long as there are contrasting features present
* __*FeatureMap 3*__ appears to have picked up horizontal edges while __*FeatureMap 4*__ has picked up vertical lines
* __*FeatureMap 1*__ & __*FeatureMap 5*__ has picked up 45 degrees lines going in either direction
* however, I am not quite sure, what __*FeatureMap 0*__ is picking up as it looks remarkably similar to __*FeatureMap 3*__ but less pronounced.

Restrospectively, it would be interesting to increase the number of filters in the first layer and see if it would improve the accuracy of the model.

![alt text][cnn_visualized]


[//]: # (Image References)

[image1]: ./docs/histogram_dataset.png "Visualization"
[image1_1]: ./docs/InputData.png "Training Data"
[image1_2]: ./docs/RGB1.png "RGB Image 1"
[image2]: ./docs/NormalizedData.png "Histogram Equalized"

[images_mean]: ./docs/MeanImages.png "Images close to mean"
[images_2500]: ./docs/MoreImages.png "Images 2500"

[image5_1]: ./docs/new_images/GermanSign1.jpeg "End of all speed and passing limits"
[image5_2]: ./docs/new_images/GermanSign2.png  "Go straight or left"
[image5_3]: ./docs/new_images/GermanSign3.png "Speed limit (30km/h)"
[image5_4]: ./docs/new_images/GermanSign4.png "General caution"
[image5_5]: ./docs/new_images/GermanSign5.png "Speed limit (70km/h)"
[cnn_visualized]: ./docs/VisualizedCNN.png "Visualized CNN"
[softmax]: ./docs/Softmax.png "Softmax Top 5"
[training_epochs]: ./docs/FinalTraining.png "Final training"
[validation_epochs]: ./docs/FinalValidation.png "Final Validation"
