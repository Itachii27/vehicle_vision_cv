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

    # Object Detection (Pedestrians, Cars, Signs) with NMS
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]

    confidence_thresh = 0.4
    nms_iou_thresh = 0.5

    filtered = [d for d in detections if d[4] >= confidence_thresh]
    annotated = frame.copy()

    if len(filtered) > 0:
        boxes = np.array([d[:4] for d in filtered], dtype=np.float32)
        scores = np.array([d[4] for d in filtered])
        classes = [int(d[5]) for d in filtered]

        # Convert to format expected by NMSBoxes
        bboxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            bboxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

        indices = cv2.dnn.NMSBoxes(
            bboxes=bboxes,
            scores=scores.tolist(),
            score_threshold=confidence_thresh,
            nms_threshold=nms_iou_thresh
        )

        # Soft colors
        label_colors = {
        'car': (255, 255, 255),          # White
        'person': (50, 205, 50),         # Green
        'truck': (0, 0, 255),            # Red
        'bus': (255, 0, 255),            # Magenta
        'traffic light': (0, 255, 255),  # Yellow
        'stop sign': (128, 0, 128),      # Purple
        }

        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x, y, w, h = bboxes[i]
            x1, y1, x2, y2 = x, y, x + w, y + h
            label = results.names[classes[i]]
            color = label_colors.get(label, (200, 200, 200))  # Default: gray
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {scores[i]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Lane detection (white threshold in HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([255, 80, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=50)

    if lines is not None:
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if 30 < abs(angle) < 150 and min(y1, y2) > frame.shape[0] // 2:
                filtered_lines.append((x1, y1, x2, y2))

        drawn = []
        for x1, y1, x2, y2 in filtered_lines:
            similar = False
            for dx1, dy1, dx2, dy2 in drawn:
                if abs(x1 - dx1) < 10 and abs(x2 - dx2) < 10:
                    similar = True
                    break
            if not similar:
                drawn.append((x1, y1, x2, y2))
                # Use subtle cyan for lane lines
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 140, 255), 2)

    cv2.imshow('Vehicle Vision System', annotated)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
