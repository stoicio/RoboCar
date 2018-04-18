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

TODO Sample Image
TODO Sample Non Car

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

TODO Sample HOG Images for Car
TODO Sample HOG Image for NOn Car

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
* This resulted in a total of 144 combinations. `extract_features.get_feature_vectors()` method was used to extract the feature vector for both car & non car datasets using each of these 144 combinations
* Each extracted feature set was saved to HDF5 file for later retireval evaluation using a classifier. [Code for saving and loading feature sets](./vehicle_detector/utils/dataset.py)

### Classifier & Hyperparameter Exploration:
* Scaler
* GridSearch
* Final Model Parameters
* Save pickled model
### Predictions using the classifier

#### Sliding windows & Multiscale detection

#### Heatmaps & Thresholding



[//]: # (Image References)

[SampleCarImages]: ./docs/car-samples.png "SampleCars"
[SampleNonCarImages]: ./docs/non-car-samples.png "SampleNonCars"
