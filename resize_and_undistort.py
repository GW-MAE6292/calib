import numpy as np
import cv2
import os
import sys


def undistort_and_resize(file_name, K, dist_coeffs, resize_scale=1.0, \
        output_file_name='output'):
    '''
    Undistort and resize a video, and save it as a new video.

    Args:
        file_name: [string] Name of the input video file to be resized
        K: [3x3 numpy array] Camera matrix for the input video
        dist_coeffs: [mx1 numpy array] Distortion coefficients
        resize_scale: [float] Resize scale, default is 1.0
        output_file_name: [string] Name of the output video file, default is 
            'output'
    '''

    # Check the size of the camera matrix.
    m, n = np.shape(K)
    assert m == 3, 'K must be a 3 by 3 array'
    assert n == 3, 'K must be a 3 by 3 array'

    # Check if the file exists.
    if not os.path.isfile(file_name):
        raise Exception('Input video file {} not found'.format(file_name))


    # Create a VideoCapture object and read from input file.
    cap = cv2.VideoCapture(file_name)

    # Get metadata from the input video.
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    # Find the optimal camera matrix and image size.
    K_optimal, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, \
        (w, h))
    x, y, w, h = roi


    # Open the VideoWriter object for saving the undistorted and resized image.
    w_scaled = int(w * resize_scale)
    h_scaled = int(h * resize_scale)
    
    os_platform = sys.platform

    output_file_name = output_file_name + '.avi'
    if os_platform == "linux" or os_platform == "linux2":
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    elif os_platform == "darwin":
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    elif os_platform == "win32":
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    else:
        raise Exception('Your OS is {}, which is not supported.\n'
            'Please update the FourCC for your OS'.format(os_platform))

    out = cv2.VideoWriter(output_file_name, fourcc, fps, (w_scaled, h_scaled))


    i_frame = 0

    # Read until video is completed
    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        

        # Undistorting
        undistorted = frame.copy()
        undistorted = cv2.undistort(undistorted, K, dist_coeffs, None, \
            K_optimal)

        # Cropping the image
        undistorted = undistorted[y:+y+h, x:x+w]

        # Resize
        resized = undistorted.copy()
        resized = cv2.resize(resized, (w_scaled, h_scaled))

        # Display the resulting frames
        cv2.imshow('Frame', frame)
        cv2.imshow('Undistorted', undistorted)
        cv2.imshow('Resized', resized)

        out.write(resized)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i_frame += 1
        

    # When everything done, release
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def scale_camera_matrix(camera_matrix, scale):
    '''
    Rescale the camera matrix.

    Once you scale the size of a video, the values that you should use for the 
    camera matrix must be scaled. This function scales the camera matrix.

    Args:
        camera_matrix: [3x3 numpy array] Camera matrix to be scaled
        scale: [float] scaling factor
    
    Returns:
        camera_matrix_scaled: [3x3 numpy array] Scaled camera matrix
    '''

    camera_matrix_scaled = scale * camera_matrix
    camera_matrix_scaled[2, 2] = 1.0
    return camera_matrix_scaled



if __name__ == '__main__':

    file_name = 'IMG_0062.MOV'

    # Define camera matrix.
    K = np.array([[1.6837083872223784e+03, 0.0, 9.8044354416223359e+02],
                  [0.0, 1.6761085098363696e+03, 5.4978445539786844e+02],
                  [0.0, 0.0, 1.0]])

    # Define distortion coefficients.
    dist = np.array([2.0984248955697157e-01, -1.1297655035589924e+00,
                     1.0011742063016877e-02, 6.7864190733829145e-03,
                     2.4092908061807035e+00])

    undistort_and_resize(file_name, K, dist)