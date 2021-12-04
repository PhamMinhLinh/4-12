import numpy as np
import cv2
ip='rtsp://admin:123456@192.168.1.2:8554/profile0'
cap = cv2.VideoCapture(ip)

while(cap.isOpened()):
    ret, frame = cap.read()


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()