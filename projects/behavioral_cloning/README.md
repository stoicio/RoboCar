
## Behavioral Cloning - Predict steering angle for a simulated car
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal of this project was to build Neural Network model that would be used to autonomously drive a car in a simulator, by predicting steering angle given an image from the car's viewport

### Submission - List of Files:

1. `model.h5` - Trained weights
2. `navigator` module - which contains all the helper files
3. `model.py` which is the glue code that ties all the helper functions together and trains the module.
4. `drive.py` Used along with the `model.h5` to predict the steering angles for images incoming from the simulator.
5. `output_video.mp4` a screen grab of the car being driven autonomously in the simulator using the trained model and the weights.
5. `output_video_reverse.mp4` a screen grab of the car being driven autonomously in the simulator's track 1 in the opposite direction.
6. `README.md` - Write of the project - You are reading it !


### Model Architecture

For this project, I decided to use a network based on [Nvidia's paper](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) which seems to have performed well in this problem domain. I also tried out [CommaAI's](https://github.com/commaai/research/blob/master/train_steering_model.py) steering model, but the nvidia's model outperformed it and was much less complex (Nivida model had 24X less parameters compared to CommaAI's)
The network architecture, which consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers is shown below.
![Model][Model]

#### Model features - [Code](./navigator/utils/models.py):

* The network starts off with a Lambda layer, that is used to perfrom image normalization on the input image. This layer takes an input image and transforms the data to be zero centered - which helps in training the network faster.
    - As an added benefit, the normalization scheme can be changed with the network architecture,if desired and also let it be accelerated via GPU Processing.
* Strided convolutions are used in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution(strides=1) with a 3×3 kernel size in the final two convolutional layers.
* Convolutional layers are followed by 3 fully connected layers and final output node, which predicts the steering angles in radians.
* Weights regularization on both the convolution and dense layers were added to prevent overfitting
* Used `RELU` activation function in all layers (except the last layer) to introduce non-linearity. 
* The network weights are trained to minimized the mean-squared error between the steering value output by the network and the steering angles recorded in the simulator.


Layer (type)           |      Output Shape        |    Param #   
----------------------|--------------------------|--------------
lambda_1 (Lambda)      |      (None, 66, 200, 3)  |      0         
conv2d_1 (Conv2D)      |      (None, 31, 98, 24)  |      1824      
conv2d_2 (Conv2D)      |      (None, 14, 47, 36)  |      21636     
conv2d_3 (Conv2D)      |      (None, 5, 22, 48)   |      43248     
conv2d_4 (Conv2D)      |      (None, 3, 20, 64)   |      27712     
conv2d_5 (Conv2D)      |      (None, 1, 18, 64)   |      36928     
flatten_1 (Flatten)    |      (None, 1152)        |      0         
dense_1 (Dense)        |      (None, 100)         |      115300    
dense_2 (Dense)        |      (None, 50)          |      5050      
dense_3 (Dense)        |      (None, 10)          |      510       
dense_4 (Dense)        |      (None, 1)           |      11        
|                       | **Total params:**        | **252,219**                         

#### Hyperparameter Tuning:

* **Learning Rate** : Even though I used the Adam optimizer the default learning rate of 0.001 was too high, which was evident from the oscillating validation loss during the model training phase. This was later reduced to 0.0001 which while slow, consistently lowered the validation loss at each epoch
* **Regularization** : Even though the original Nvidia model didn't use any regularization methods, with limited data available to us it was important not to overfit. Dropout layers at different convolution and fully connected stages were tried. This didn't help improving the model's performance. After some research & help in the course forums, I settled upon using the kernel (weight) regularizers provided in Keras with a value 0.001
* **Camera Offset / Recovery Distance** - Although not a network related model, this played an important part in data augmentation. The details of this are described in the __Data Augmentation Section #2 Code block__

### Training Data:
Data collection was easily the most challenging part of this project ! While I first started with the data published by Udacity, quickly realized that the bias to overcome driving straight was going to be hard. To this effect I collected additional training data, recording only when driving from the edges to the centre. This proved unviable as well since the bias towards driving straight was an aspect of the track itself. The picture below shows the histogram of sampled data.

#### Raw Dataset - Highly Skewed:

![UnbalancedData][UnbalancedData]

#### Data Augmentation:
To overcome this I decided to generate training data artificially by augmenting the already collected training data. The steps below describes each of the augmentation done & link to the relevant sections of code.

##### 1. Minimizing Zero Angles [Code - `remove_zero_angle_logs` Line #82-94](./navigator/utils/randomized_augmentation.py):
* Since the data was highly skewed to driving straight, any augmentation that was done on this would have the same skewedness. To avoid this, an utility function was written that would randomly pick only a percentage of data with zero steering angles.
    - Random deletion of these data was done for each epoch of training, so that the same images won't get deleted every time & the network could see different sets of images with zero angles.
##### 2. Using images from left & right cameras & angle correction:
* The simulator at each time step also records images collected from the left & right side of the vehicle similar to the center camera. For example, the following image contains the position in the track from three cameras
    ![Original][Original]
* But since the steering angles are recorded only for the center position, we should augment that value if we are to use these two images. Fortunately, we can approximately estimate this value. 
    - [Code `choose_cam_postion` Line #61](./navigator/utils/randomized_augmentation.py) randomly picks one of the three camera angles.
    - [Code `choose_cam_postion` Line #42](./navigator/utils/randomized_augmentation.py) adjusts the steering angle for the picked image using the calculation shown below. 
```python
# Given:
#  S - current steering angle to reach the center of the track
# Assumptions:
#  Offset - distance from center camera to camera's on either edge.
#         - On average for sedans & SUVs this can be approximated to be 1 METER
#  Recovery Distance - distance within which the vehicle should recover back to the center of the lane
#                    - this is "tunable" parameter. Since the recovery distance also depends on speed, 
#                      its hard to estimate this during training time. If we choose a value too low, 
#                      the steering behaviour might be aggressive, if we choose a value too high the vehicle
#                      might not revcover on time and drive off the lane. I chose this to be 10 METERS to 
#                      allow for recovery time of 2 seconds when driving at 20 mph
# TO FIND: S`
# tan(S`) = tan(S) + (RecoveryDistance / Offset) 
# Approximating tan(X) to X
# S` = S + (RecoveryDistance / Offset)
```
##### 3. Random Gamma Correction (Brightness)[Code `random_gamma_adjust` Line #97](./navigator/utils/randomized_augmentation.py) :
* Randomly increase or decrease the brightness of the image, by adjusting the value of each pixel by `Out = In * (1/gamma)` where gamma is selected from a range [0.3, 2.0] randomly.

##### 3. Horizontal Flip [Code `h_flip` Line #7](./navigator/utils/randomized_augmentation.py):
* About 50% of the time, randomly flip an image abouts its vertical axis and adjust the steering angle accordingly
##### 4. Random Translation: [Code `random_translate` Line #15](./navigator/utils/randomized_augmentation.py):
* Randomly translate an image in the x & y axis within a given range. 

Each image in the training data is put through some combinations of the this augmentations using the [method `read_and_augment_image` (Line #105)](./navigator/utils/randomized_augmentation.py). 

#### Image Generator: [Code `batch_generator` Line #35](./navigator/utils/dataset.py):
* To enable this augmentation on the fly and to leverage keras's `fit_generator` method, a python generator was used. Given a batch_size the generator can infinitely generate training (or validation) samples of that batch size.
* Steps performed by the generator in **Training**:
    1. Loads all the training data.
    1. Filters out data with zero steering angles (retains only 5% of the data)
    1. Reads the image from disk and augments the image as described above.
    1. Prep the image to be fed into the model
        - Crops part of the image - removing horizon and hood of the car
        - Converts the image to YUV Color Space
        - Resizes the image to (200, 66) compatible with the model definied above. (NOTE this size is not set, and can be modified if required)
* Steps performed by generator in **Validation mode**:
    1. Reads the image from disk. Only center camera postion images are used for validation purpose.
    1. Preps the image to be fed in the model - (Step 4 from above). No Augmentation is done on these images.

#### Dataset distribution after augmentation
As an effort of this augmentation, the training images are now well balanced & are more uniformly distributed as it can be seen below (Data collected by recording steering angles during training phase.)

![BalancedData][BalancedData]

Some examples of different kinds of augmentations are tabulated below.

Augmentation   | Left Camera, Center Camera and Right Camera|
---------------|--------------------------------------------|
Original       | ![Original][Original]                      |
Steering angle | ![SteeringCorrection][SteeringCorrection]  |
Brightness     | ![GammaAdjust][GammaAdjust]                |
Horizontal Flip| ![HFlips][HFlips]                          |
Translation    | ![Translate][Translate]                    |
**All Mixed**    | ![Together][Together]                    |
**ModelInput**    | ![ModelInput][ModelInput]                    |

#### Training Startegy [Code](./model.py):

Some important aspects implemented to optimize training & speed up iterations

1. Pandas dataframe was used to read the CSV log file and quickly apply transformations (changing file paths, extracting required columns etc). `Line #48-50`
2. sklearn utils were used to randomly shuffle and divide the available data into training and validation sets. I used a split of 0.8 / 0.2 for training and validation. No test set was kept separate since the simulator was a better test for the model.
3. Keras callbacks were leveraged for the following purposes
    - Validation loss was not really a good indicator of how well the training was performing. (this was resolved to an extent when I reduced the initial learning rate for the Adam optimizer). So I saved a model at the end of each epoch by using `ModelCheckpoint` callbacks `Line #59`. Model with the lowest validation loss was used to test the model performace in the simulator
    - Training process was stopped early, if there was no significant reduction in the validation loss. This was achieved by using `EarlyStopping` callbacks `Line #60`

#### Video Output:

* [Video Track 1 Forward](https://youtu.be/TsrghfTBbVQ)
* [Video Track 1 Reverse](https://youtu.be/H3kUd1jDl-M)

#### Challenges & Further Work:

As described earlier, this project was good exercise on collecting and augmenting data when data that is good to use is hard to come by. I am quite certain I could have spent countless more hours tweaking data collection and augmentation methods.

In real life scenarios where we have access to hard measurements, some assumptions made here (recovery distance, camera offset) can be determined accurately. We could also apply geometry based computer vision techniques to gather more data that might be useful in augmenting the training data. For example, instead of minimizing the squared error between the steering angles, a different loss function could potentially be _calculating the steering angle required to reach the lane center in a set amount of time_. (I believe this would alleviate some problems that comes with data collection, but it would outside the domain of behavioral cloning)

In the near future, I want to make the model robust enough to generalize to other tracks not seen before in training data & potentialy use the throttle and brake data from the simulator to control all three output vectors of a Self Driving Car!

[//]: # (Image References)

[BalancedData]: ./docs/BalancedData.png "BalancedData"
[UnbalancedData]: ./docs/UnbalancedData.png "UnbalancedData"


[Original]: ./docs/Original.png "Original"
[SteeringCorrection]: ./docs/SteeringCorrection.png "SteeringCorrection"
[GammaAdjust]: ./docs/GammaAdjust.png "GammaAdjust"
[Translate]: ./docs/Translate.png "Translate"
[HFlips]: ./docs/HFlips.png "HFlips"
[Together]: ./docs/Together.png "Together"
[ModelInput]: ./docs/ModelInput.png "ModelInput"
[Model]: ./docs/model.png "Model"
