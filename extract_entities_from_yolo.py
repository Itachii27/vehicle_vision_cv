import cv2
import torch
import numpy as np
import json  # <-- new

# Load YOLOv5 (from PyTorch Hub or local path)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

def get_yolo_detections(frame, conf_threshold=0.4):
    results = model(frame)
    detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, class]
    entities = []

    for *box, conf, cls in detections:
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        entity_crop = frame[y1:y2, x1:x2]
        entities.append({
            'label': label,
            'bbox': [x1, y1, x2, y2],
            'confidence': float(conf)
        })

    return entities

def visualize_detections(frame, entities, output_path='yolo_output.jpg'):
    for ent in entities:
        x1, y1, x2, y2 = ent['bbox']
        label = f"{ent['label']} ({ent['confidence']:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    cv2.imwrite(output_path, frame)
    print(f"[✓] Detections visualized and saved to {output_path}")
    cv2.imshow("YOLO Detections", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    frame = cv2.imread("test_driving_scene.png")

    if frame is None:
        print("❌ Image not found. Make sure 'test_driving_scene.png' exists in the folder.")
        exit()

    entities = get_yolo_detections(frame)

    with open("detected_entities.json", "w") as f:
        json.dump(entities, f, indent=2)
    print("[✓] Detections saved to detected_entities.json")

    visualize_detections(frame.copy(), entities)
