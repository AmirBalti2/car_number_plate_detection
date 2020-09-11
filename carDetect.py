import numpy as np 
import matplotlib.pyplot as plt 
import  cv2 as cv

cardetect=cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_russian_plate_number.xml')

cap=cv.VideoCapture("car.mp4")
cap.set(cv.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,80)
if (cap.isOpened()==False):
    print("error")
while True:
    ret,frame=cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cars=cardetect.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(25,25))
    for (x,y,w,h) in cars:
        cv.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        plate_detect=frame[y:y+h,x:x+w]
        imgBlur=cv.GaussianBlur(plate_detect,(23,23),30)
        frame[y:y+imgBlur.shape[0],x:x+imgBlur.shape[1]]=imgBlur
    if ret == True:
        cv.imshow('video',frame)
    if cv.waitKey(0) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()

