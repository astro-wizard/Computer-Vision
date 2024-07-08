import cv2
import numpy as np

# Camera 1 parameters
camera_matrix1 = np.array([[894.12028268, 0., 391.37973472],
                           [0., 899.42562872, 131.86418221],
                           [0., 0., 1.]])
dist_coeffs1 = np.array([0.05961526, 0.56832833, -0.03232476, 0.01113443, -8.60597234])

# Camera 2 parameters
camera_matrix2 = np.array([[1.09858230e+03, 0.00000000e+00, 2.88016122e+02],
                           [0.00000000e+00, 1.10085098e+03, 1.76485803e+02],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeffs2 = np.array([5.11293243e-01, -9.70982657e+00, -2.24921207e-02, -9.08004253e-04, 4.63900694e+01])

# Stereo parameters
R = np.array([[0.99034542, -0.01804487, 0.13744209],
              [0.02399138, 0.99884073, -0.04173255],
              [-0.1365297, 0.04462706, 0.98963027]])
T = np.array([[-3.93287338], [0.09934257], [1.89644675]])

# Initialize cameras
camera1 = cv2.VideoCapture(0)
camera2 = cv2.VideoCapture(1)

if not camera1.isOpened() or not camera2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# Capture frames and perform undistortion
while True:
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()

    if not ret1 or not ret2:
        print("Error: Could not read frame from one or both cameras.")
        break

    # Undistort frames
    frame1_undistorted = cv2.undistort(frame1, camera_matrix1, dist_coeffs1)
    frame2_undistorted = cv2.undistort(frame2, camera_matrix2, dist_coeffs2)

    # Convert frames to grayscale (necessary for stereoRectify)
    gray1 = cv2.cvtColor(frame1_undistorted, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_undistorted, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    h, w = gray1.shape[:2]

    # Compute rectification transforms for each camera
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(camera_matrix1, dist_coeffs1,
                                                camera_matrix2, dist_coeffs2,
                                                (w, h), R, T)

    # Compute the undistortion and rectification transformation map
    map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix1, dist_coeffs1, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix2, dist_coeffs2, R2, P2, (w, h), cv2.CV_32FC1)

    # Apply the transformation
    rectified1 = cv2.remap(frame1_undistorted, map1x, map1y, cv2.INTER_LINEAR)
    rectified2 = cv2.remap(frame2_undistorted, map2x, map2y, cv2.INTER_LINEAR)

    # Stitch images together
    stitcher = cv2.Stitcher_create()
    result, stitched_frame = stitcher.stitch([rectified1, rectified2])

    if result == cv2.Stitcher_OK:
        cv2.imshow('Stitched View', stitched_frame)
    else:
        print("Error during stitching. Error code:", result)
        combined_frame = cv2.hconcat([rectified1, rectified2])
        cv2.imshow('Combined View (fallback)', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera1.release()
camera2.release()
cv2.destroyAllWindows()
