from ultralytics import YOLO
import cv2

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")

# Load image
image = cv2.imread(r"C:\Users\hp\Desktop\ai_self_billing_demo\images\sample.jpg")
if image is None:
    print("❌ Image not loaded. Check file path.")
    exit()


# Run detection
results = model(image)

detected_products = []

PRODUCT_DATABASE = {
    "bottle": {"id": 101, "price": 30},
    "banana": {"id": 102, "price": 10},
    "apple": {"id": 103, "price": 20},
    "cup": {"id": 104, "price": 50},
    "tea packet": {"id": 105, "price": 70},
    "detergent": {"id": 106, "price": 150},
    "orange": {"id": 107, "price": 95},
    "corn flakes": {"id": 108, "price": 170}
    
}

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in PRODUCT_DATABASE:
            product = PRODUCT_DATABASE[label]
            detected_products.append({
                "name": label,
                "id": product["id"],
                "price": product["price"]
            })

print(detected_products)

# Show detection image
annotated = results[0].plot()
cv2.imshow("AI Detection Output", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
