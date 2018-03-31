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

I started by wrting a utitlity class to compute the camera matrix & distortion coefficients. [Link to code](./lane_detector/camera_utils.py). `__calibrate_camera() - Line #54` method does most of the heavy lifting, while few other helpers functions allows easier options for debugging & undistorting iamges

##### Finding Corners
* The class is initialized by passing in the **number of internal corners** in the chess board along X & Y directions & the path to images of the chessboar in different orientations and positions.
* A list of points representing the chessboard corners in 3D space was created & `cv2.findChessboardCorners` was used to locate the corners in pixel space. 
    - Finding chessboard corners in 3 images failed - calibration1.jpg, calibration3.jpg, calibration5.jpg. In `calibration1.jpg` & `calibration5.jpg` I suspect these two images failed since one of the corners were too close to the edge of the image & hence that particular corner might have been difficult to pick up by the method.
    - `calibration3.jpg` appeared to suffer because of lighting conditions (top row has a glare, while bootom rows has better contrast). Looking up the documentation for the function, I found the `cv2.CALIB_CB_ADAPTIVE_THRESH` which does adaptive thresholding rather than fixed thresholds to convert to a bindary image. This helped in identifying the corners.
* Once the corners were located, `cv2.cornerSubPix` was used to refine the corner points. The **refinement process was terminated when either 30 iterations were completed or when change between two iterations were less than 0.001** whichever came first.
* if `store_output_images` flag was set during intialization, the chessboard corners were annotated and written to the disk.
* With the corner points located in 3d space and pixel space, `cv2.calibrateCamera` was used to calculate the camera matrix & distortion coefficients.

##### Undistortion
* `undistort_image() - Line #131` is a helper function that given a image or a path to image undistorts the image using the calculated camera matrix & distortion coefficients.
![Calibration3][Calib1]
![Calibration2][Calib2]
* One of the Undistorted test images can be seen below. The distortion effect is obvious near the car's hood and the right edge of the image.
![CarTest1][CarCalib1]


[//]: # (Image References)

[Calib1]: ./output_images/camera_calib/calibration3.png "Calibration3"
[Calib2]: ./output_images/camera_calib/calibration2.png "Calibration2"
[CarCalib1]: ./output_images/camera_calib/test1.png "CarTest1"
