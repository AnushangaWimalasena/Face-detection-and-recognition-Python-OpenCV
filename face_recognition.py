import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

while(1):

    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.5, minNeighbors = 5)

    for(x,y,w,h) in face:

        roi_gray = frame_gray[y:y+h, x:x+h]
        img_face = 'face-img.png'
        cv2.imwrite(img_face, roi_gray)

        color = (255,0,0)
        stroke = 2
        cv2.rectangle(frame, (x,y),(x+w, y+h), color, stroke)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
