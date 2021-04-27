# Camera Calibration and Video Undistortion



## Calibration

Camera calibration is to determine the intrinsic paramter $K$ and the lens distortion coefficients. In class, we discussed using a checker board image to estimate $K$. 

Variaous checkerboard images are availble at

https://markhedleyjones.com/projects/calibration-checkerboard-collection

The following tutorial covers the detailed examples for calibration using multiple checkerboard images.

https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html



## Undistortion and Rescale 

Once the camera is calibrated, each video frame should be undistorted. Also, it can be rescaled if needed. But, when the image is rescaled, the instrinsic parameter should be rescaled in a consistent way. 

The attached code `resize_and_undistort.py` include several functions to undistort and resize a video, written by Kani. You are welcome to use this. But, this is provided as is, without additional supports. 

