import cv2
import numpy as np
import json

# Load image
image_path = 'test_driving_scene.jpg'
image = cv2.imread("test_driving_scene.png")
height, width, _ = image.shape

# Load detected entities from previous step
with open('detected_entities.json', 'r') as f:
    detections = json.load(f)

# Risk weights (customize based on domain knowledge)
risk_weights = {
    'person': 1.0,
    'bicycle': 0.9,
    'car': 0.7,
    'motorcycle': 0.8,
    'bus': 0.6,
    'truck': 0.5,
    'traffic light': 0.3,
    'stop sign': 0.4
}

# Create a blank heatmap
heatmap = np.zeros((height, width), dtype=np.float32)

# Gaussian parameters
sigma = 50

# Generate heatmap based on detections
for det in detections:
    label = det['label']
    x1, y1, x2, y2 = map(int, det['bbox'])
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    risk_score = risk_weights.get(label, 0.2)

    # Create a 2D Gaussian blob
    blob = np.zeros((height, width), dtype=np.float32)
    cv2.circle(blob, (cx, cy), sigma, risk_score, -1)
    blob = cv2.GaussianBlur(blob, (0, 0), sigmaX=sigma, sigmaY=sigma)

    heatmap = np.maximum(heatmap, blob)  # Take maximum risk across overlapping

# Normalize heatmap
heatmap = np.clip(heatmap / heatmap.max(), 0, 1)
heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

# Overlay heatmap on image
overlay = cv2.addWeighted(image, 0.7, heatmap_color, 0.5, 0)

# Save and show
cv2.imwrite('risk_heatmap_overlay.jpg', overlay)
print("✅ Risk heatmap overlay saved as 'risk_heatmap_overlay.jpg'")
cv2.imshow("Risk Heatmap", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()