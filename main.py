#from turtle import tracer
import cv2
#import tracker  
from tracker import *

# create tracker object
tracker = EuclideanDistTracker()


cap = cv2.VideoCapture("4k Video of Highway Traffic!.mp4")

# Object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    #print(height, width)


    # Extract region of interest
    roi = frame[140: 800,200: 1200]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 1000:
         #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2) 
         x, y, w, h = cv2.boundingRect(cnt)
         cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

         detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
       x, y, w, h, id = box_id
       cv2.putText(roi, str(id), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
       cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    print(boxes_ids)
        # print(x, y, w, h)

    print(detections)
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    Key = cv2.waitKey(10)
    if Key == 7 :
        break

cap.release()
cv2.destroyAllWindows()