import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture('videos/6.mp4')
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=2, circle_radius=2)

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, 
                    mpFaceMesh.FACEMESH_CONTOURS, drawSpec, 
                    drawSpec)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 135), 
        cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 10)

    # resize img
    resized_img = cv2.resize(img, (640, 480))

    cv2.imshow("Image", resized_img)
    cv2.waitKey(1)
