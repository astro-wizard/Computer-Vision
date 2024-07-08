import cv2
import numpy as np

def initialize_camera(index):
    """
    Initialize a camera given its index.

    Parameters:
    index (int): The index of the camera to initialize.

    Returns:
    cv2.VideoCapture: The initialized camera object.

    Raises:
    Exception: If the camera could not be opened.
    """
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        raise Exception(f"Error: Could not open camera {index}.")
    return cam

def capture_images(cam1, cam2, num_images, pattern_size):
    """
    Capture images from two cameras and find chessboard corners.

    Parameters:
    cam1 (cv2.VideoCapture): The first camera.
    cam2 (cv2.VideoCapture): The second camera.
    num_images (int): Number of images to capture for calibration.
    pattern_size (tuple): The size of the chessboard pattern (columns, rows).

    Returns:
    tuple: A tuple containing:
        - objpoints (list): 3D points in real-world space.
        - imgpoints1 (list): 2D points in image plane for camera 1.
        - imgpoints2 (list): 2D points in image plane for camera 2.
    """
    objpoints = []  # 3D points in real-world space
    imgpoints1 = []  # 2D points in image plane for camera 1
    imgpoints2 = []  # 2D points in image plane for camera 2

    # Prepare object points like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

    for i in range(num_images):
        # Read frames from both cameras
        ret1, img1 = cam1.read()
        ret2, img2 = cam2.read()

        if not ret1 or not ret2:
            print(f"Error: Could not read frame {i+1}.")
            continue

        # Convert frames to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners in the images
        ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)

        if ret1 and ret2:
            # If corners are found, add object points and image points
            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)

            # Draw chessboard corners on the images
            cv2.drawChessboardCorners(img1, pattern_size, corners1, ret1)
            cv2.drawChessboardCorners(img2, pattern_size, corners2, ret2)

            # Display the images with drawn corners
            cv2.imshow('Camera 1', img1)
            cv2.imshow('Camera 2', img2)
            cv2.waitKey(500)  # Wait for 500 milliseconds

    return objpoints, imgpoints1, imgpoints2

def calibrate_cameras(objpoints, imgpoints1, imgpoints2, gray_shape):
    """
    Calibrate two cameras and perform stereo calibration.

    Parameters:
    objpoints (list): 3D points in real-world space.
    imgpoints1 (list): 2D points in image plane for camera 1.
    imgpoints2 (list): 2D points in image plane for camera 2.
    gray_shape (tuple): Shape of the grayscale image.

    Returns:
    tuple: A tuple containing:
        - mtx1 (np.ndarray): Camera matrix for camera 1.
        - dist1 (np.ndarray): Distortion coefficients for camera 1.
        - mtx2 (np.ndarray): Camera matrix for camera 2.
        - dist2 (np.ndarray): Distortion coefficients for camera 2.
        - R (np.ndarray): Rotation matrix between the cameras.
        - T (np.ndarray): Translation vector between the cameras.
    """
    # Calibrate camera 1
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, gray_shape, None, None)
    # Calibrate camera 2
    ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray_shape, None, None)

    # Stereo calibration to get rotation and translation between the cameras
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints1,
        imgpoints2,
        mtx1,
        dist1,
        mtx2,
        dist2,
        gray_shape,
        criteria=criteria,
        flags=flags
    )

    return mtx1, dist1, mtx2, dist2, R, T

def main():
    """
    Main function to perform camera initialization, image capture, and calibration.
    """
    # Initialize both cameras
    cam1 = initialize_camera(0)
    cam2 = initialize_camera(1)

    num_images = 20  # Number of images to capture for calibration
    pattern_size = (9, 6)  # Size of the chessboard pattern

    # Capture images from both cameras
    objpoints, imgpoints1, imgpoints2 = capture_images(cam1, cam2, num_images, pattern_size)

    if len(objpoints) < num_images:
        print(f"Error: Only {len(objpoints)} valid image pairs captured. Calibration might not be accurate.")
        return

    # Get the shape of the grayscale image
    gray_shape = cv2.cvtColor(cam1.read()[1], cv2.COLOR_BGR2GRAY).shape[::-1]
    # Calibrate the cameras
    mtx1, dist1, mtx2, dist2, R, T = calibrate_cameras(objpoints, imgpoints1, imgpoints2, gray_shape)

    # Print calibration results
    print("Camera 1 matrix: \n", mtx1)
    print("Camera 1 distortion coefficients: \n", dist1)
    print("Camera 2 matrix: \n", mtx2)
    print("Camera 2 distortion coefficients: \n", dist2)
    print("Rotation matrix: \n", R)
    print("Translation vector: \n", T)

    # Release cameras and close all OpenCV windows
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

    print("Calibration completed.")

if __name__ == "__main__":
    main()
