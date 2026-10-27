from flask import Flask, jsonify, render_template
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# 🔹 ABSOLUTE PATH TO IMAGE (CHANGE ONLY IF YOUR PATH IS DIFFERENT)
IMAGE_PATH = r"C:\Users\hp\Desktop\ai_self_billing_demo\images\sample3.jpg"

# 🔹 Load AI model ONCE
model = YOLO("yolov8n.pt")

# 🔹 Product database
PRODUCT_DATABASE = {
    "bottle": {"id": 101, "price": 30},
    "banana": {"id": 102, "price": 10},
    "apple": {"id": 103, "price": 20},
    "cup": {"id": 104, "price": 50},
    "orange": {"id": 105, "price": 15},
    "corn flakes": {"id": 108, "price": 170},
    "eggs": {"id": 109, "price": 45},
    "milk": {"id": 110, "price": 25}


}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect")
def detect():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        return jsonify({"error": "Image not found at path", "path": IMAGE_PATH})

    results = model(image)
    detected_items = []
    seen = set()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in PRODUCT_DATABASE and label not in seen:
                seen.add(label)
                product = PRODUCT_DATABASE[label]
                detected_items.append({
                    "name": label,
                    "id": product["id"],
                    "price": product["price"]
                })

    return jsonify(detected_items)

if __name__ == "__main__":
    app.run(debug=True)
