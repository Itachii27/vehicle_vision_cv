import json
import numpy as np

# Simulated MLP using weighted heuristics instead of training
def compute_attention_score(entity, image_center):
    x1, y1, x2, y2 = entity['bbox']
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    center_dist = np.linalg.norm(np.array([cx, cy]) - np.array(image_center))

    # Normalize distance (lower is better)
    norm_dist = center_dist / (image_center[0] + image_center[1])

    # Class importance (higher = more attention)
    class_weights = {
        'person': 1.0,
        'car': 0.8,
        'motorcycle': 0.7,
        'truck': 0.6,
        'traffic light': 0.5,
        'bus': 0.5,
        'bicycle': 0.6,
        'stop sign': 0.4
    }
    class_weight = class_weights.get(entity['label'], 0.3)

    confidence = entity['confidence']  # from YOLO

    # Weighted sum to simulate attention score
    attention_score = (
        0.5 * class_weight +
        0.3 * confidence +
        0.2 * (1 - norm_dist)  # closer to center = more attention
    )
    return round(attention_score, 3)

# Main logic
if __name__ == "__main__":
    with open('detected_entities.json', 'r') as f:
        entities = json.load(f)

    # Image size (used previously)
    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 720
    center = (IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2)

    for ent in entities:
        ent['attention_score'] = compute_attention_score(ent, center)

    # Save results
    with open('attention_scores.json', 'w') as f:
        json.dump(entities, f, indent=2)

    print("[✓] Attention scores computed and saved to attention_scores.json")
