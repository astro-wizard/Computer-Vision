# Stereo Camera Calibration

This repository contains a Python script for calibrating a stereo camera system using OpenCV. The script captures images from two cameras, detects chessboard corners, and performs stereo calibration to estimate the intrinsic and extrinsic parameters of the cameras.

## Code Overview

The code is organized into five functions:

### 1. `initialize_camera(index)`
Initializes a camera given its index. Returns a `cv2.VideoCapture` object if the camera is successfully opened, otherwise raises an exception.

### 2. `capture_images(cam1, cam2, num_images, pattern_size)`
Captures images from two cameras and detects chessboard corners. Returns three lists: `objpoints` (3D points in real-world space), `imgpoints1` (2D points in image plane for camera 1), and `imgpoints2` (2D points in image plane for camera 2).

### 3. `calibrate_cameras(objpoints, imgpoints1, imgpoints2, gray_shape)`
Performs stereo calibration using the captured images. Returns the intrinsic and extrinsic parameters of the cameras: `mtx1`, `dist1`, `mtx2`, `dist2`, `R`, and `T`.

### 4. `main()`
The main function that initializes the cameras, captures images, and performs stereo calibration.

## How to Use

1. Connect two cameras to your system and ensure they are properly configured.
2. Run the script using Python (e.g., `python stereo_calibration.py`).
3. The script will capture images from both cameras and perform stereo calibration.
4. The calibration results will be printed to the console.

## Note

- This script assumes a chessboard pattern with a size of 9x6 is used for calibration.
- The script uses the `cv2.findChessboardCorners` function to detect chessboard corners, which may not work well with low-quality images or complex backgrounds.
- The calibration results may not be accurate if the cameras are not properly synchronized or if the images are not captured simultaneously.

## Dependencies

- OpenCV (version 4.x or later)
- NumPy

## License

This script is released under the MIT License.

## Code Explanation

The code is written in Python and uses the OpenCV library for image processing and computer vision tasks. The script consists of four main functions:

- `initialize_camera`: Initializes a camera given its index.
- `capture_images`: Captures images from two cameras and detects chessboard corners.
- `calibrate_cameras`: Performs stereo calibration using the captured images.
- `main`: The main function that initializes the cameras, captures images, and performs stereo calibration.

The script uses the following OpenCV functions:

- `cv2.VideoCapture`: Initializes a camera.
- `cv2.read`: Reads a frame from a camera.
- `cv2.cvtColor`: Converts an image from one color space to another.
- `cv2.findChessboardCorners`: Detects chessboard corners in an image.
- `cv2.drawChessboardCorners`: Draws chessboard corners on an image.
- `cv2.calibrateCamera`: Calibrates a camera using a set of images.
- `cv2.stereoCalibrate`: Performs stereo calibration using a set of images from two cameras.

The script also uses NumPy for numerical computations and array operations.
