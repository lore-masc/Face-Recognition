import numpy as np
import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     fgmask = fgbg.apply(frame)

#
#     cv2.imshow('frame', fgmask)
#     cv2.imshow('frame_original', frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break

panel = np.zeros([100, 700], np.uint8)
cv2.namedWindow("panel")

def nothing(x):
    pass

cv2.createTrackbar("L - h", "panel", 0, 255, nothing)
cv2.createTrackbar("U - h", "panel", 255, 255, nothing)

cv2.createTrackbar("L - s", "panel", 0, 255, nothing)
cv2.createTrackbar("U - s", "panel", 255, 255, nothing)

cv2.createTrackbar("L - v", "panel", 0, 255, nothing)
cv2.createTrackbar("U - v", "panel", 255, 255, nothing)

cv2.createTrackbar("S ROWS", "panel", 0, 480, nothing)
cv2.createTrackbar("E ROWS", "panel", 480, 480, nothing)

cv2.createTrackbar("S COLS", "panel", 0, 640, nothing)
cv2.createTrackbar("E COLS", "panel", 640, 640, nothing)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    s_r = cv2.getTrackbarPos("S ROWS", "panel")
    e_r = cv2.getTrackbarPos("E ROWS", "panel")
    s_c = cv2.getTrackbarPos("S COLS", "panel")
    e_c = cv2.getTrackbarPos("E COLS", "panel")

    roi = frame[s_r: e_r, s_c: e_c]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - h", "panel")
    u_h = cv2.getTrackbarPos("U - h", "panel")
    l_s = cv2.getTrackbarPos("L - s", "panel")
    u_s = cv2.getTrackbarPos("U - s", "panel")
    l_v = cv2.getTrackbarPos("L - v", "panel")
    u_v = cv2.getTrackbarPos("U - v", "panel")

    lower_green = np.array([l_h, l_s, l_v])
    upper_green = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(roi, roi, mask=mask)
    fg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    cv2.imshow('bg', bg)
    cv2.imshow("fg", fg)
    cv2.imshow("panel", panel)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
