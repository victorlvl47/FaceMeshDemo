import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture('videos/1.mp4')

while True:
    success, img = cap.read()

    # resize img
    resized_img = cv2.resize(img, (640, 480))

    cv2.imshow("Image", resized_img)
    cv2.waitKey(1)
