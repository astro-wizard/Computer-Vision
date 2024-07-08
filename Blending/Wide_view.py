import cv2
import numpy as np

# Initialize both cameras
camera1 = cv2.VideoCapture(0)  # Change the index if the camera is not the default camera
camera2 = cv2.VideoCapture(1)  # Change the index to match your second camera

if not camera1.isOpened() or not camera2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# ORB detector
orb = cv2.ORB_create()

# FLANN based matcher
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  # 2
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()

    if not ret1 or not ret2:
        print("Error: Could not read frame from one or both cameras.")
        break

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match descriptors
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    if len(good_matches) > 10:
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp frame1 to frame2's perspective
        height, width = frame2.shape[:2]
        frame1_warped = cv2.warpPerspective(frame1, H, (width, height))

        # Blend the images
        blend_mask = np.ones_like(frame2, dtype=np.float32)
        frame1_warped = frame1_warped.astype(np.float32)
        frame2 = frame2.astype(np.float32)

        blend_mask_warped = cv2.warpPerspective(blend_mask, H, (width, height))
        alpha = blend_mask_warped[:, :, 0:1]

        blended = alpha * frame1_warped + (1 - alpha) * frame2
        blended = blended.astype(np.uint8)

        # Show the blended image
        cv2.imshow('Wide View', blended)
    else:
        print("Not enough matches found - {}/{}".format(len(good_matches), 10))
        combined_frame = cv2.hconcat([frame1, frame2])
        cv2.imshow('Combined View (fallback)', combined_frame)

    # Display the original views from both cameras
    cv2.imshow('Camera 1 View', frame1)
    cv2.imshow('Camera 2 View', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera1.release()
camera2.release()
cv2.destroyAllWindows()
