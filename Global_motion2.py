import cv2 as cv
import numpy as np
import sys
#O que fazer?
#Percorrer o dataset de imagens
#salvar os resultados em um csv

def write_result ():
    





def Motion_features (img):
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # The video feed is read in as a VideoCapture object
    cap = cv.VideoCapture(img)
    color = (0, 255, 0)
    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, first_frame = cap.read()
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
    prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    mask = np.zeros_like(first_frame)

    while(cap.isOpened()):
        prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

    #ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates sparse optical flow by Lucas-Kanade method
        next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        # Selects good feature points for previous position
        good_old = prev[status == 1]
        # Selects good feature points for next position
        good_new = next[status == 1]
        # Draws the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), color, 2)
            frame = cv.circle(frame, (a, b), 3, color, -1)
        # Overlays the optical flow tracks on the original frame
        prev_gray = gray.copy()
        prev = good_new.reshape(-1, 1, 2)

    if cv.waitKey(10) & 0xFF == ord('q'):
        sys.exit()

    # The following frees up resources and closes all windows 
    cap.release()
    cv.destroyAllWindows()