import cv2
import torch
import numpy as np
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture('media/road_scene.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object Detection (Pedestrians, Cars, Signs)
    results = model(frame)
    annotated = np.squeeze(results.render())

    # Road & Lane Markings using white threshold
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([255, 80, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Detect lines with Hough Transform
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=50)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.imshow('Vehicle Vision System', annotated)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
