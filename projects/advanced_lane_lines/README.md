## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal of this project is to write software pipeline to identify the lane boundaries in a video from a front-facing camera on a car.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Camera calibration:

I started by wrting a utitlity class to compute the camera matrix & distortion coefficients. [Link to code](./lane_detector/utils/camera_utils.py). `__calibrate_camera() - Line #87` method does most of the heavy lifting, while few other helpers functions allows easier options for debugging & undistorting iamges

##### Finding Corners
* The class is initialized by passing in the **number of internal corners** in the chess board along X & Y directions & the path to images of the chessboar in different orientations and positions.
* A list of points representing the chessboard corners in 3D space was created & `cv2.findChessboardCorners` was used to locate the corners in pixel space. 
    - Finding chessboard corners in 3 images failed - calibration1.jpg, calibration3.jpg, calibration5.jpg. In `calibration1.jpg` & `calibration5.jpg` I suspect these two images failed since one of the corners were too close to the edge of the image & hence that particular corner might have been difficult to pick up by the method.
    - `calibration3.jpg` appeared to suffer because of lighting conditions (top row has a glare, while bootom rows has better contrast). Looking up the documentation for the function, I found the `cv2.CALIB_CB_ADAPTIVE_THRESH` which does adaptive thresholding rather than fixed thresholds to convert to a bindary image. This helped in identifying the corners.
* Once the corners were located, `cv2.cornerSubPix` was used to refine the corner points. The **refinement process was terminated when either 30 iterations were completed or when change between two iterations were less than 0.001** whichever came first.
* if `store_output_images` flag was set during intialization, the chessboard corners were annotated and written to the disk.
* With the corner points located in 3d space and pixel space, `cv2.calibrateCamera` was used to calculate the camera matrix & distortion coefficients.

##### Undistortion
* `undistort_image() - Line #164` is a helper function that given a image or a path to image undistorts the image using the calculated camera matrix & distortion coefficients.
![Calibration3][Calib1]
![Calibration2][Calib2]
* One of the Undistorted test images can be seen below. The distortion effect is obvious near the car's hood and the right edge of the image.
![CarTest1][CarCalib1]


### Perspective Transformation: [Link to code](./lane_detector/utils/perspective_transformer.py)

`PerspectiveTransform` class takes in a set of source and destination points and calculates a homography matrix which used to map points from the source coordinate system to the destination coordinate system. While the points used in this class are hardcoded for the purposes of this project, the class can be used in a more generalized manner by passing in any source and destination points when intializing the object.

##### Points to get Bird's eye view:
* The main goal of the performing perspective transform in this project is to **transform the camera's lane image to a overhead view**. Another way of saying it is, we want **to reproject the lane lines such that they are parallel to each other** as they are in the real world, instead of lines that meet at the horizon
* This can be done by finding a homography matrix. The points I used to calculate the mapping from one view to the other are derived by the code snippet
    ``` python
    offset_pix = 100  # Padding around the lane lines for the warped image

    y_bottom = 700  # Lowest part of the image where lane lines can be seen
    y_horizon = 470 # Horizon point beyond which little or no lanes can be seen
    n_cols = 1280  # hardcoded image size for Udacity Project

    self.src_pts = np.array([[offset_pix, y_bottom],  # Bottom Left
                            [n_cols - offset_pix, y_bottom],  # Bottom Right
                            [n_cols / 2 + offset_pix, y_horizon],  # Top right
                            [n_cols / 2 - offset_pix, y_horizon]  # Top Left
                            ], dtype=np.float32)

    dst_n_rows, dst_n_cols = 450, 800
    self.dst_pts = np.array([[offset_pix, dst_n_rows],  # Bottom Left
                            [dst_n_cols - offset_pix, dst_n_rows],  # Bottom Right
                            [dst_n_cols - offset_pix, 0],  # Top Right
                            [offset_pix, 0],  # Top Left
                            ], dtype=np.float32)

    ```
* This gives us the following sets of points.

    | Source       | Destination   | Location     |
    |:------------:|:-------------:|:------------:| 
    | 100, 700     | 100, 450      | Bottom Left  |
    | 1180, 700    | 700, 450      | Bottom Right |
    | 740, 470     | 700, 0        | Top Right    |
    | 540,470      | 100, 0        | Top Left     |
* Using these points, `cv2.getPerspectiveTransform` method is used to calculate a homography matrix. Further a reverse mapping homography image is also calculated to unwarp the image. 
* Two helper functions `PerspectiveTransform.warp_image` and `PerspectiveTransform.unwarp_image` is used in the final pipeline to facilitate this. Some example warped images can be seen below.

    |          Original Image      |           Warped Image           |
    |:----------------------------:|:--------------------------------:|
    | ![LineSample1][pt_line_1_in] | ![LineSample1Out][pt_line_1_out] | 
    | ![LineSample2][pt_line_2_in] | ![LineSample2Out][pt_line_2_out] | 


### Extracting the lane lines: [Link to code](./lane_detector/utils/lane_detector.py)
* Extracting the lane lines by choosing a combination of color channel thresholds, morphological operations, gradients was the most time consuming part of the project for me ! I found thresholding while trivial to be very tricky to generalize to all the images in the video. 
* I tried various operations & experiments some of them being converting image to YUV & LAB color space, canny edge detector, laplacian operator, adaptive thresholding of the image, sobel operators. I also tried various [morphological operations](https://docs.opencv.org/2.4/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html) to reduce noise, close out hollow lanes etc. Most of these experiments can be found in the [BinaryImage noteboook](./notebooks/BinaryImage.ipynb)
* I also found out the usage of [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html) very useful in determining various thresholds & allevated some difficulty in manually changing the threshold values
##### Final Thresholding pipeline
* In the end, I decided to settle on a simple thresholding function, that worked well for the project video. I observed that **S & V Channels in the HSV color space best picked out both the yellow and white lane lines**. While V channel picked up the yellow lines, V picked up both yellow and white to an resonable extent. 
* To **enhance the white lane pixels, I created a grayscale image, by stacking up S value in one channel and V values in 2 channels and averaging them out**.
* The gray scale image is thresholded to create a binary image, with white pixels representing the lane lines.
* this operation can be found in the `LaneDetector.combine_s_v` method in line #29. LaneDetector is the utility class which runs the pipeline on the image recieved from a video file.


|  Original Image     |   S Image           |       V Channel      | Binary Thresholded      |
|:-------------------:|:-------------------:|:--------------------:|:-----------------------:|
| ![Test1C][test_1_c] | ![Test1S][test_1_s] |  ![Test1V][test_1_v] | ![Test1Bin][test_1_bin] |
| ![Test2C][test_2_c] | ![Test2S][test_2_s] |  ![Test2V][test_2_v] | ![Test2Bin][test_2_bin] |
| ![Test3C][test_3_c] | ![Test3S][test_3_s] |  ![Test3V][test_3_v] | ![Test3Bin][test_3_bin] |


### Polynomial Fitting: [Link to code](./lane_detector/utils/lane_line.py)
Once a binary image is obtained with the lane lines isolated, we need to obtain a parameteric representation of the line by determining a line that best fits the lane pixels. `class LaneLine` in the code linked above does most of the heavy lifting, It receives as input the binary image obtained by thresholding and returns the following items

1. Array of points in pixel space forming a line that represents the left lane & right lane separately
2. Radius of curvature of the lane
3. The vehicles current offset from the centre of the lane.

##### Steps followed
1. Each column in the bottom half binary image is added up to find the column that has the most number of white pixels. Searching to the either side of the image centre gives us the prospective positions of the left and the right lane starting postions - `Line #43-47 in LaneLine.__get_lane_pixels()`
2. Once the prospective base positions of the two lanes are found, a region of interest is formed around the base positions and projected along the vertical direction to get all possible pixels belonging to that lane. The indices of these pixels are stored. - `Line #50-63 in LaneLine.__get_lane_pixels()`
3. A 2nd degree polynomial fit is obtained using `np.polyfit`. These polynomial coefficients are then used to find a best fit x-coordinate for each y-coordinate. This gives us the lane line. `Line #101 - LaneLine.__get_best_fit_line()`. 
4. **Radius of curvature :** The identified pixel coordinates are converted to real world coordinates by scaling them by a factor of
    ```
    Y_MTRS_PER_PIXEL = 30 / 720  # meters per pixel in y direction
    X_MTRS_PER_PIXEL = 3.7 / 700  # meters per pixel in x direction
    ```
    A new set of polynomial coefficients are determined by fitting a 2nd degree polynomial to these real world coordinates. Using the new polynomial coeffiecients the radius of curvature is determined at the lowest visible y-coordinate. Radius for both the left and right lane lines are determined & averaged out to give the final curvature measurement. `Line #67 LaneLine.__calculate_radii_of_curvature()`
5. **Lane Offset :** Offset from the centre of the lane is found by getting the pixel difference between the centre of lane (average of left and right lanes) & the image centre and scaling them by a factor of `X_MTRS_PER_PIXEL = 3.7/700`. `Line 106 - LaneLine.__calculate_lane_offset()`

Example images of lane extraction process:

|  Warped Image     |   Binary Image        | Histogram of white pixels| Line Fit            |
|:-------------------:|:-------------------:|:--------------------:|:-----------------------:|
| ![Lane1C][lane-1-c] | ![Lane1B][lane-1-b] |  ![Lane1H][lane-1-h] | ![Lane1D][lane-1-d] |
| ![Lane1C][lane-2-c] | ![Lane1B][lane-2-b] |  ![Lane1H][lane-2-h] | ![Lane1D][lane-2-d] |
| ![Lane1C][lane-3-c] | ![Lane1B][lane-3-b] |  ![Lane1H][lane-3-h] | ![Lane1D][lane-3-d] |


### Pipeline: [Link to Code](./lane_detector/utils/lane_detector.py)
`class LaneDetector` in the code linked above ties everything together. It can be used to process either a single frame of image or a stream of images from a video clip by setting the flag`process_stream=True | False` when creatign the object. All the processing is done in `LaneDetector.process_frame()`. The steps in the pipeline are as follows.
1. Create LaneDetector object, that takes as input `CameraCalibration` object & `PerspectiveTransform` object (both described above). if the flag `process_stream=True` a couple of additional steps takes place in `LaneLine.extract_lane_lines` that helps smooth out the detections from frame to frame.
2. `LaneDetector.process_frame` takes in a single color image. The camera calibration object is used to undistort this image by calling `CameraCalibration.undistort_image()`
3. The undistorted_image is warped & thresholded using the `PerspectiveTransform.warp_image` & `LaneDetector.combine_s_v` repectively. We have a binary image at this point with lane pixels in white.
4. The binary image is processed by `LaneLine.extract_lane_lines()` method which returns a parameterized left & right lane, along with the curvature & lane offset values
5. `LaneDetector.draw_lane_lines()` uses these measurements and draws the lines on an image using `cv2.polylines & cv2.fillPoly`. In addition the radius of curvature and lane offset are also annotated on the frame using `cv2.putText`
6. The warped image is projected back to the original frame using `PerspectiveTransform.unwarp_image` and blended with the original color image to annotate it lane markings.

##### Smoothing.
if `process_stream` flag is set to true, `LaneLine.extract_lane_lines() #line 128-161` does [weighted moving average](https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average) over the last 4 frames in the sequence. This helps smooth the frame to frame transitions by discounting the importance of the calulcated values in the most recent frame & giving some importance to the previous frame. Emprically the moving average weights I decided to use were as follows (`t being the most recent frame`). 

|Time|Weights|
|:--:|:--:|
|t   |0.8|
|t-1| 0.6|
|t-2| 0.4|
|t-3| 0.2|
|t-4| 0.1|

## Link to Pipeline Video - [P4 Video](https://youtu.be/Jg5iAH10AVk)

## Example images with lanes drawn
|  Warped Image     |   Binary Image        |
|:-------------------:|:-------------------:|
| ![OLane1C][o-lane-1] | ![FLane1][f-lane-1] |
| ![OLane1C][o-lane-2] | ![FLane1][f-lane-2] |
| ![OLane1C][o-lane-3] | ![FLane1][f-lane-3] |

### Discussions:

#### Perspective Transformation Improvements:
* The homography matrix is only calculated once before the start of the pipeline & uses hardcoded values. This only works as long as the points are carefully chosen & pavement is flat throughout the entirety of the video. the matrix calculated this way will not perform well under changes pavement elevation : for eg, downhill roads, uphill roads etc
* A more robust way of doing this would be, rather than choosing the points manually, use hough lines and / or vanishing points to calculate the where the lane lines meet and choose the points backtracking from there. This can be done every few frames or use a smoothing function which retains the values from the last few frame. Smoothing function can be reset, if the euclidean distance between the matrices of frames of the pipeline goes above a certain threshold

#### Color Thresholding:
* While extracting the lane pixels with the simple thresholding I used here, works well for the project_video it fails in numerous frames in the challenge videos (low contrast, cracks in pavement, drastict shadows). As I had said this was the most troublesome part of the project to me. 
* But the code is modular enough that I can experiment with other techniques and quickly iterate on this part of the pipeline which could lead to better results on the harder videos. I definitely want to experiment with edge detector & hough lines to boostrap the lane detection process and supplement it with the color thresholding options.

#### Performance:
* Another question that occured to me, for which I quite haven't figured out the answer to. In highway driving, how performant should the pipeline be - so that vehicle can quickly react to changes in lane curvature. This tells me that cutting of lane detection's region of interest at half the image's height wouldn't help much.

I also looked around for reasearch materials on this topic and there seems to be a lot of *pre deep learning* techniques which include vanishing points, navigating on roads without lane markings etc. 

Another important dataset that I came across was [TuSimple dataset](http://benchmark.tusimple.ai/#/t/1) which has labelled images in various driving conditions. I look forward to experimenting with that soon ! 


[//]: # (Image References)

[Calib1]: ./doc/calibration/calibration3.png "Calibration3"
[Calib2]: ./doc/calibration/calibration2.png "Calibration2"
[CarCalib1]: ./doc/calibration/test1.png "CarTest1"
[Curvature]: ./doc/curvature.png "curvature"

[pt_line_1_in]: ./doc/perspective_transform/line_1_in.png "LineSample1"
[pt_line_2_in]: ./doc/perspective_transform/line_2_in.png "LineSample2"
[pt_line_1_out]: ./doc/perspective_transform/line_1_out.png "LineSample1Out"
[pt_line_2_out]: ./doc/perspective_transform/line_2_out.png "LineSample2Out"

[test_1_c]: ./doc/binary_images/test_1_c.png "Test1C"
[test_1_s]: ./doc/binary_images/test_1_s.png "Test1S"
[test_1_v]: ./doc/binary_images/test_1_v.png "Test1V"
[test_1_bin]: ./doc/binary_images/test_1_bin.png "Test1Bin"

[test_2_c]: ./doc/binary_images/test_2_c.png "Test2C"
[test_2_s]: ./doc/binary_images/test_2_s.png "Test2S"
[test_2_v]: ./doc/binary_images/test_2_v.png "Test2V"
[test_2_bin]: ./doc/binary_images/test_2_bin.png "Test2Bin"

[test_3_c]: ./doc/binary_images/test_3_c.png "Test3C"
[test_3_s]: ./doc/binary_images/test_3_s.png "Test3S"
[test_3_v]: ./doc/binary_images/test_3_v.png "Test3V"
[test_3_bin]: ./doc/binary_images/test_3_bin.png "Test3Bin"

[lane-1-c]: ./doc/extracted_lane/lane-1-c.png "Lane1C"
[lane-2-c]: ./doc/extracted_lane/lane-2-c.png "Lane2C"
[lane-3-c]: ./doc/extracted_lane/lane-3-c.png "Lane3C"

[lane-1-d]: ./doc/extracted_lane/lane-1-d.png "Lane1D"
[lane-2-d]: ./doc/extracted_lane/lane-2-d.png "Lane2D"
[lane-3-d]: ./doc/extracted_lane/lane-3-d.png "Lane3D"

[lane-1-h]: ./doc/extracted_lane/lane-1-h.png "Lane1H"
[lane-2-h]: ./doc/extracted_lane/lane-2-h.png "Lane2H"
[lane-3-h]: ./doc/extracted_lane/lane-3-h.png "Lane3H"

[lane-1-b]: ./doc/extracted_lane/lane-1-b.png "Lane1B"
[lane-2-b]: ./doc/extracted_lane/lane-2-b.png "Lane2B"
[lane-3-b]: ./doc/extracted_lane/lane-3-b.png "Lane3B"

[lane-1-f]: ./doc/extracted_lane/lane-1-f.png "Lane1F"
[lane-2-f]: ./doc/extracted_lane/lane-2-f.png "Lane2F"
[lane-3-f]: ./doc/extracted_lane/lane-3-f.png "Lane3F"

[f-lane-1]: ./doc/final_image/f-lane-1.jpg "FLane1"
[f-lane-2]: ./doc/final_image/f-lane-2.jpg "FLane2"
[f-lane-3]: ./doc/final_image/f-lane-3.jpg "FLane3"

[o-lane-1]: ./doc/final_image/o-lane-1.jpg "OLane1"
[o-lane-2]: ./doc/final_image/o-lane-2.jpg "OLane2"
[o-lane-3]: ./doc/final_image/o-lane-3.jpg "OLane3"
