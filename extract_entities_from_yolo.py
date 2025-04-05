import cv2
import torch
import numpy as np

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
            'bbox': (x1, y1, x2, y2),
            'crop': entity_crop,
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
    # Test with a sample image
    frame = cv2.imread("test_driving_scene.png")
    entities = get_yolo_detections(frame)
    visualize_detections(frame.copy(), entities)