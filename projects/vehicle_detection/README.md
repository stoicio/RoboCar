## Vehicle Detection Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal of this project is to write a software pipeline to detect vehicles in a video using a classical machine learning approach.

The steps taken to achieve the final end result are the following

* Extract postive & negative image samples from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/) datasets
* Filter out similar images from GTI dataset since they are extracted from a video sequence. Some images
might be too similar to each other affecting the classifier performance
* Build an image descriptor (HOG & Color Historgram) & extract feature vectors for the given dataset
* Use the feature vectors to build an SVM Classifier
* Detect vehicles on each frame of the video, by searching the entire image using sliding windows at different scales
* Aggregate the detections from different scales by buidling a heatmap of detections and thresholding them

* Draw bouding boxes around the thresholded heatmaps / detections


### Dataset Exploration - [Link to Notebook](./notebooks/PrepData.ipynb):
As a first step, all the car and non car datasets were aggregated and their locations were written to a json file
to make it easier to process them. No preprocessing was required, since all the images were cropped to the same size.
* Two different json files were created. One containing the path to all the images in the dataset and another with similar images filtered out.

##### Filtering Similar Image : [Cell #3 in notebook](./notebooks/PrepData.ipynb)

* Since the GTI dataset contains images extracted from a video sequence some images were too similar to each other. This would have been redundant & could have overfit the classifier for those samples. To avoid this the following similar images were removed by comparing their histograms
* `calcHist` &  `compareHist` methods from [OpenCV](https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html) were used to calculate the histograms of all three channels of each image & any image with an intersection score of 750 compared to an already selected image was rejected. 
* Since the positive and negative examples were resonably well balance, furthere data augmentation was not done at this point.

    | Dataset                 | Postive Samples   | Negative Sample     |
    |:-----------------------:|:-----------------:|:-------------------:| 
    | GTI                     | 2826              | 3900                |
    | KTTI                    | 5966              | N/A                 |
    | GTI(Filtered)           | 1720              | 3900                |
    | Extras                  | N/A               | 5068                |
    |**Total**                |  **8792**         |   **8968**          | 
    |**Total (Filtered)**     |  **7686**         | **8968**            | 

##### Sample Postive Images ![SampleCars][SampleCarImages]
##### Sample Negative Images ![SampleNonCars][SampleNonCarImages]

### Image Descriptors:
After some initial exploration I decided to use two image descriptors. Color Histogram - to get the color information embeded in the image, Histogram of Oriented Gradients (HOG) to determine the extract the spatial features in the image (edges, orientation of edges which roughly defines the outline of the car in positive samples)

#### Color Histogram - [Code](./vehicle_detector/descriptors/color.py): 

* This image descriptor gives us a basic color & intensity distribution of the image
* Histograms were calculated for every channel in the image & combined to return single feature vector.

#### Sample Color Features :

| Image               | Features                       |
|---------------------|--------------------------------|
| ![car][car]         |![car-color][car-color]         |
| ![non-car][non-car] |![non-car-color][non-car-color] |


#### Histogram of Oriented Gradients [Code](./vehicle_detector/descriptors/hog.py): 

The HOG features are widely use for object detection. HOG decomposes an image into small squared cells, computes an histogram of oriented gradients in each cell, normalizes the result using a block-wise pattern, and return a descriptor for each cell.

A good overview of this technique can be found in the seminal paper by [Dalal & Triggs](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)- Reading through this paper also gives a general sense of parameter tuning for HOG Features described below

The three main parameters of HOG descriptor are 
* Number of Orientation Bins
* Number of Pixels Per Cell
* Number of Cells Per Block across which normalization has to be carried out.

In addition to this images in different color space could produce different results as well. For example, GRAYSCALE image would have a lower number of hog features when compared to RGB image, but might be less robust.

The robustness of the descriptor depends on the combinations of these parameters & finding out those combinations manually was very time consuming. So an automated process was put in place so that different combinations of the above parameters could be experimented with and the best performing one could be chosen.(This process is described in the next section). The final image descriptor parameters that were chosen after experimentation

| Parameter       | Value   |
|-----------------|---------|
|Orientation Bins |12       |
|Pixels Per Cell  |(16, 16) |
|Cells per Pixel  |(4, 4)   |
|Color Historgram | YES     |
|Image ColorSpace | YCrCb   |

| Image               | Features                       |
|---------------------|--------------------------------|
| ![car][car]         |![car-hog][car-hog]         |
| ![non-car][non-car] |![non-car-hog][non-car-hog] |

### Feature Extraction [Code](./vehicle_detector/extract_features.py):
*  To explore the different combinations of feature descriptors, I listed out different options for each parameter and generated all possible combinations between them using the following code snippet.

```python
import itertools

opts_color_hist = [True, False]
opts_orients = [9, 10, 12]
opts_n_pixels= [(8,8), (12,12), (16,16)]
opts_n_cells = [(2,2), (4,4)]
opts_cspace = ['BGR2GRAY', 'BGR', 'BGR2HSV', 'BGR2YCrCb']

# Get all possible combinations of the above options
parameter_combinations = list(itertools.product(opts_color_hist, opts_orients, opts_n_pixels,
                                           opts_n_cells, opts_cspace))
```
* This resulted in a total of 144 combinations. `extract_features.get_feature_vectors()` method was used to extract the feature vector for both car & non car datasets using each of these 144 combinations. ( Only a subset of the images extracted earlier were used. 2000 random images were sampled from both the positive and negative image sets) 
    * _Some of these combinations were later rejected, because of  pixels, cells count being incompatible with the sliding window size used to detect cars_. If the pixel & cell count doesn't divide the window size evenly, then the HOG descriptor cannot cover the entire region of the image.
* Each extracted feature set was saved to HDF5 file for later retireval evaluation using a classifier. [Code for saving and loading feature sets](./vehicle_detector/utils/dataset.py)


### Classifier & Hyperparameter Exploration:

#### Support Vector Machine:
Support vector machine was used as the classifier for this project. [Scikit-Learn's](http://scikit-learn.org/stable/modules/svm.html) SVM Implementation was used. The two main parameters for the SVM classifier are 

* Penalty Parameter (C) - Defines the smoothness of the decision boundary
* Kernel Coefficient (gamma) - Defines how much influence a single training example has
To Determine a proper choice of `C` and `gamma` [`sklearn's GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) was used.

##### Training the model [Notebook](./notebooks/TrainClassifier.ipynb):
The SVM model was trained in two phases.

1. Models were trained using different combinations of feature sets (extracted from 2000 random samples of each class) & different parameters for the SVM using GridSearch. Each model's accuracy was also calculated. These parameters can be found in this [JSON File](./docs/parameter_combinations.json)
2. In Phase 2, the top 10 models were chosen from the previous experiment & were retrained on the full dataset with the parameters obtained above. The relative stats between these paramters are shown below. (all values were normalized to a scale of 1 to 10 for easier interpretation) [JSON FILE](./docs/top_models.json)
    ![Top10Models][Top10Models]

Hog_Color_Space | Color_Hist | Pixels_Per_Cell | Orientations | Cells_Per_Block | Gamma | Accuracy |
--------------- | ---------- | --------------- | ------------ | --------------- | ----- | -------- |
BGR2HSV         | true       | 8,8             | 10           | 4,4             | auto  | 99.199   |
**BGR2YCrCb**   | **true**   | **16,16**       | **12**       | **4,4**         | **auto**  | **99.545** |
BGR2YCrCb       | true       | 8,8             | 10           | 4,4             | auto  | 99.272   |
BGR2HSV         | false      | 16,16           | 10           | 2,2             | auto  | 98.854   |
BGR2YCrCb       | true       | 16,16           | 9            | 4,4             | auto  | 99.691   |
BGR             | true       | 8,8             | 9            | 4,4             | auto  | 98.581   |
BGR2YCrCb       | true       | 16,16           | 10           | 4,4             | auto  | 99.582   |
BGR2YCrCb       | false      | 16,16           | 9            | 2,2             | auto  | 99.036   |
BGR2HSV         | true       | 16,16           | 12           | 2,2             | auto  | 99.254   |
BGR             | true       | 16,16           | 10           | 4,4             | auto  | 99.145   |
BGR2YCrCb       | true       | 8,8             | 9            | 2,2             | auto  | 99.108   |

3. Out of these 10 Models, the best performing model and its parameters were used for the final classifier. This classifier & the feature scaler (used to normalize the feature vectors) were saved for later retrieval & use in the detection pipeline
    * the models were also trained with `probability=True` so that predictions will also have the probability along with it using [`svm.predict_proba`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict_proba). This gives us a thresholding option to eliminated False Positives later in the pipeline.
4. Final Model's parameters:
    
    | Parameter       | Value   |
    |-----------------|---------|
    |Orientation Bins |12       |
    |Pixels Per Cell  |(16, 16) |
    |Cells per Pixel  |(4, 4)   |
    |Color Historgram | YES     |
    |Image ColorSpace | YCrCb   |
    |Gamma            | auto    |
    |C                | 10      |
    |**TestAccuracy** | **99.573%** |

### Predictions using the classifier [Code](./vehicle_detector/object_detection/object_detector.py):

Now that a viable model is trained, we use it to detect the location of cars in the test_images. The `ObjectDetector` class linked above encapsulates the steps below. The main methods in the class are 

* `find_cars` - Given an image and a sliding window scale, extracts patches of images and checks if that patch contains a vehicle. Since HOG descriptors are NOT scale invariant, the feature vectors are extracted at different scales so that the vehicles can be identified irrespective of the relative position of the car from the camera
    1. A `min_prob` option can also be given to reject detections that are less than this probability
    2. This function scales the boundarys of the image patches in which cars were detected and returns a list of bounding boxes.
* `add_heat` - Given a list of bounding boxes of detected cars at various scales, creates a heatmap of the bounding boxes. areas which has many overlapping boxes will be brighter indicating high probability of a vehicle being present. an optional `threshold` parameter can also be passed to eliminate potential false positives which has very few overlapping bounding boxes

* `draw_labelled_boxes` - Given the heatmap image calculated in the above step, seperate blobs are identified using `scipy.ndimage` [library](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html). A bounding box is draw encompassing each blob of labelled data.

### Ouput Images:
| Multi Scale Detection       |  Heatmap             | Bounded Vehicles |
|-----------------------------|----------------------|------------------|
| ![msd1][msd1]               | ![heat1][heat1]      | ![box1][box1]    |
| ![msd2][msd2]               | ![heat2][heat2]      | ![box2][box2]    |
| ![msd3][msd3]               | ![heat3][heat3]      | ![box3][box3]    |
| ![msd4][msd4]               | ![heat4][heat4]      | ![box4][box4]    |
| ![msd5][msd5]               | ![heat5][heat5]      | ![box5][box5]    |
| ![msd6][msd6]               | ![heat6][heat6]      | ![box6][box6]    |

### Project Video: [Link](https://youtu.be/1r_h6gTV2ZU)
* The video sequence is processed using the same pipeline as the still image - expect that the heatmaps of frames within the last 1 second is retained and averaged out before drawing bounding boxes.

### Challenges & Further Work:

This project gave me a good insight into building a traditional machine learning pipeline & the challenges involved in it. Feature engineering was especially tedious and time consuming and in sharp contrast to current deep learning pipelines, where the feature engineering is almost a black box.

Another challenge was to get the pipeline to work in real time. This pipeline takes about 1.2 seconds per frame on average. Multiscale detection is very time consuming but unavoidable using this pipeline. This could potentially be optimized by doing the following things
* Scan only areas where we expect new vehicles to appear from.
* Once a vehicle is detected, actively track it and extrapolate thier position from previous frames there by further restricting the area we need to do perfom multiscale detection on.

The Pipleline performs resonably well on the project video, but loses the cars when they are relatively far away fromt the camera. While I was able to detect those cars by scaling the search window by factor less than 1.0, the processing time almost doubled.

As always, It would be interesting to reimplement the project using the some now famous deep learning architectures like SSD & YOLO.

[//]: # (Image References)

[SampleCarImages]: ./docs/car-samples.png "SampleCars"
[SampleNonCarImages]: ./docs/non-car-samples.png "SampleNonCars"
[Top10Models]: ./docs/top_10_models.png "Top10Models"

[msd1]: ./docs/output/msd1.png "msd1"
[msd2]: ./docs/output/msd2.png "msd2"
[msd3]: ./docs/output/msd3.png "msd3"
[msd4]: ./docs/output/msd4.png "msd4"
[msd5]: ./docs/output/msd5.png "msd5"
[msd6]: ./docs/output/msd6.png "msd6"

[heat1]: ./docs/output/heat1.png "heat1"
[heat2]: ./docs/output/heat2.png "heat2"
[heat3]: ./docs/output/heat3.png "heat3"
[heat4]: ./docs/output/heat4.png "heat4"
[heat5]: ./docs/output/heat5.png "heat5"
[heat6]: ./docs/output/heat6.png "heat6"

[box1]: ./docs/output/box1.jpg "box1"
[box2]: ./docs/output/box2.jpg "box2"
[box3]: ./docs/output/box3.jpg "box3"
[box4]: ./docs/output/box4.jpg "box4"
[box5]: ./docs/output/box5.jpg "box5"
[box6]: ./docs/output/box6.jpg "box6"

[car]: ./docs/features/car.png "car"
[car-color]: ./docs/features/car_color.png "car-color"
[car-hog]: ./docs/features/car_hog.png "car-hog"
[non-car]: ./docs/features/non_car.png "non-car"
[non-car-color]: ./docs/features/non_car_color.png "non-car-color"
[non-car-hog]: ./docs/features/non_car_hog.png "non-car-hog"
