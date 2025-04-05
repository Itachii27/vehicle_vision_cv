import cv2
import json

def attention_to_color(score):
    """Map attention score [0, 1] to RGB color (Green → Yellow → Red)."""
    if score < 0.33:
        return (0, 255, 0)      # Green
    elif score < 0.66:
        return (0, 255, 255)    # Yellow
    else:
        return (0, 0, 255)      # Red

if __name__ == "__main__":
    image_path = "test_driving_scene.png"
    image = cv2.imread(image_path)

    with open('attention_scores.json', 'r') as f:
        entities = json.load(f)

    for ent in entities:
        x1, y1, x2, y2 = map(int, ent['bbox'])
        score = ent['attention_score']
        color = attention_to_color(score)
        label = f"{ent['label']} ({score:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_path = "risk_attention_overlay.png"
    cv2.imwrite(out_path, image)
    print(f"[✓] Risk heatmap overlay saved to {out_path}")
    cv2.imshow("Risk Heatmap", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
