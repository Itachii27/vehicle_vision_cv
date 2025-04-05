import cv2
import torch
import numpy as np
import os

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
            base_color = label_colors.get(label, (200, 200, 200))

            # -------------------- Risk Scoring --------------------
            risk_score = compute_risk_score(x1, y1, x2, y2, label, frame.shape[1], frame.shape[0])

            if risk_score > 0.6:
                risk_color = (0, 0, 255)  # Red
            elif risk_score > 0.3:
                risk_color = (0, 165, 255)  # Orange
            else:
                risk_color = (0, 255, 0)  # Green

            # -------------------- Risk Heatmap Overlay --------------------
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), risk_color, -1)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

            # Outline box and text
            cv2.rectangle(annotated, (x1, y1), (x2, y2), base_color, 2)
            cv2.putText(annotated, f"{label} {scores[i]:.2f} | Risk: {risk_score:.2f}",
                        (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 2)

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

    cv2.imshow('Vehicle Vision System', annotated)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
