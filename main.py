import os
import csv
import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture('media/road_scene.mp4')

# -------------------- Risk Score Function --------------------
def compute_risk_score(x1, y1, x2, y2, label, frame_width, frame_height):
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2
    center_offset = abs(box_center_x - frame_width / 2)
    box_area = (x2 - x1) * (y2 - y1)

    label_risk_weight = {
        'car': 1.0,
        'truck': 1.2,
        'bus': 1.5,
        'person': 2.0,
        'traffic light': 0.5,
        'stop sign': 0.3
    }

    label_weight = label_risk_weight.get(label, 0.2)
    distance_weight = max(0.1, 1 - (center_offset / (frame_width / 2)))
    size_weight = min(1.0, box_area / (frame_width * frame_height * 0.1))

    risk_score = label_weight * distance_weight * size_weight
    return risk_score

# -------------------- CSV Setup --------------------
csv_file = 'data.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Label', 'Risk Score', 'X1', 'Y1', 'X2', 'Y2'])

# -------------------- Display Settings --------------------
display_scale = 0.6  # scale down display window

important_classes = {'car', 'person', 'bus', 'truck', 'traffic light', 'stop sign'}
max_boxes = 10  # max number of bounding boxes to show

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    confidence_thresh = 0.4
    nms_iou_thresh = 0.5

    filtered = [d for d in detections if d[4] >= confidence_thresh]
    annotated = frame.copy()

    if len(filtered) > 0:
        boxes = np.array([d[:4] for d in filtered], dtype=np.float32)
        scores = np.array([d[4] for d in filtered])
        classes = [int(d[5]) for d in filtered]

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

        # Limit number of boxes
        indices = sorted(indices, key=lambda i: scores[i[0] if isinstance(i, (list, tuple, np.ndarray)) else i], reverse=True)[:max_boxes]

        label_colors = {
            'car': (255, 255, 255),
            'person': (50, 205, 50),
            'truck': (0, 0, 255),
            'bus': (255, 0, 255),
            'traffic light': (0, 255, 255),
            'stop sign': (128, 0, 128),
        }

        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x, y, w, h = bboxes[i]
            x1, y1, x2, y2 = x, y, x + w, y + h
            label = results.names[classes[i]]
            if label not in important_classes:
                continue

            base_color = label_colors.get(label, (200, 200, 200))

            # -------------------- Risk Scoring --------------------
            risk_score = compute_risk_score(x1, y1, x2, y2, label, frame.shape[1], frame.shape[0])

            # ✅ -------------------- Save to CSV --------------------
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"{timestamp:.2f}", label, f"{risk_score:.2f}", x1, y1, x2, y2])

            # -------------------- Risk Heatmap Overlay --------------------
            overlay = annotated.copy()
            if risk_score > 0.6:
                risk_color = (0, 0, 255)  # Red
            elif risk_score > 0.3:
                risk_color = (0, 165, 255)  # Orange
            else:
                risk_color = (0, 255, 0)  # Green

            cv2.rectangle(overlay, (x1, y1), (x2, y2), risk_color, -1)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), base_color, 1)
            cv2.putText(annotated, f"{label} {scores[i]:.2f} | Risk: {risk_score:.2f}",
                        (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, risk_color, 1)

    # -------------------- Lane Detection --------------------
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
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 140, 255), 2)

    # -------------------- Display Resize --------------------
    resized_frame = cv2.resize(annotated, (1280, 720))
    cv2.imshow('Vehicle Vision System', resized_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
