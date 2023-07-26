import cv2
import numpy as np

cap = cv2.VideoCapture("data/lane_test.mp4")

# Function for the trackbar
def nothing(x):
    pass

# Create a named window
cv2.namedWindow("Trackbars")

# Create the trackbars
cv2.createTrackbar("L - H", "Trackbars", 10, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 135, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 140, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    ret, frame = cap.read()

    # If the video has ended, reset the video to the first frame
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (1080,920))

    ##choosing points for perspective transformation
    tl = (322, 500)
    bl = (70, 572)
    tr = (700, 500)
    br = (1038, 570)

    cv2.circle(frame, tl, 5, (0,0,255), -1)
    cv2.circle(frame, bl, 5, (0,0,255), -1)
    cv2.circle(frame, tr, 5, (0,0,255), -1)
    cv2.circle(frame, br, 5, (0,0,255), -1)

    ##Apply perspective transformation
    pts1 = np.float32([tl,bl,tr,br])
    pts2 = np.float32([[0,0], [0,920], [1080,0], [1080, 920]])

    #Matrix to warp the image for birdseye view
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (1100, 1640))
    # Get the inverse matrix
    inverse_matrix = cv2.getPerspectiveTransform(pts2, pts1)

    ## object detection
    #Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    ##Histogram
    histogram = np.sum(mask[mask.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Generate histogram for visual representation
    histogram_image = np.zeros((300, mask.shape[1], 3), dtype='uint8')  # Change 300 to any desired height
    cv2.line(histogram_image, (midpoint, 0), (midpoint, 300), (0, 255, 0), 1)  # Green line representing midpoint
    cv2.line(histogram_image, (left_base, 0), (left_base, 300), (255, 0, 0), 1)  # Blue line representing left_base
    cv2.line(histogram_image, (right_base, 0), (right_base, 300), (0, 0, 255), 1)  # Red line representing right_base
    for i in range(1, histogram.shape[0]):
        cv2.line(histogram_image, (i-1, 300-int(histogram[i-1])), (i, 300-int(histogram[i])), (0, 255, 0), 1)

    ##Sliding window
    y = 472
    lx = []
    rx = []

    msk = mask.copy()

    while y>0:
        ##left threshold
        img = mask[y-40:y, left_base-100:left_base+100]
        contours_info = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(left_base-50 + cx)
                left_base = left_base-50 + cx

        ## Right threshold
        img = mask[y-40:y, right_base-100:right_base+100]
        contours_info = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                rx.append(right_base-50 + cx)
                right_base = right_base-50 + cx

        # Draw rectangle on the mask
        cv2.rectangle(msk, (left_base-50, y), (left_base+50, y-40), (255,255,255), 2)
        cv2.rectangle(msk, (right_base-50, y), (right_base+50, y-40), (255,255,255), 2)

        # Draw rectangles on the original frame
        rec_pts = np.float32([[left_base-50, y], [left_base+50, y-40], [right_base-50, y], [right_base+50, y-40]]).reshape(-1,1,2)
        rec_pts_transformed = cv2.perspectiveTransform(rec_pts, inverse_matrix)
        rec_pts_transformed = rec_pts_transformed.astype(int)
        cv2.rectangle(frame, tuple(rec_pts_transformed[0][0]), tuple(rec_pts_transformed[1][0]), (255,255,255), 2)
        cv2.rectangle(frame, tuple(rec_pts_transformed[2][0]), tuple(rec_pts_transformed[3][0]), (255,255,255), 2)

        y -= 40

    cv2.imshow("Frame", frame)
    cv2.imshow("BEV", transformed_frame)
    cv2.imshow("Lane Detection - Image Thresholding", mask)
    cv2.imshow("Lane Detection - Sliding Windows", msk)
    cv2.imshow("Histogram", histogram_image)

    

    if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
