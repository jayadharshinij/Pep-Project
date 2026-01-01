# Triple Riding Detection from Image (NO Helmet)

from ultralytics import YOLO
import cv2
import os

# -------------------------------
# Load YOLOv8 model (COCO)
# -------------------------------
model = YOLO("yolov8s.pt")

# -------------------------------
# Violation logic
# -------------------------------
def check_triple_riding(detected_classes):
    riders = detected_classes.count("person")
    motorcycle_present = "motorcycle" in detected_classes
    triple_riding = motorcycle_present and riders > 2
    return riders, motorcycle_present, triple_riding

# -------------------------------
# Get image path from user
# -------------------------------
IMAGE_PATH = input("Enter full image path: ").strip()

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError("Image not found. Please check the path.")

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError("Unable to read image.")

# -------------------------------
# Run YOLO detection
# -------------------------------
results = model(image, conf=0.4)

detected_classes = []

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        detected_classes.append(label)

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

# -------------------------------
# Apply violation rules
# -------------------------------
riders, motorcycle_present, triple_riding = check_triple_riding(detected_classes)

# -------------------------------
# Print results
# -------------------------------
print("\nDetected Objects:", detected_classes)
print("Number of Riders:", riders)
print("Motorcycle Detected:", motorcycle_present)

if triple_riding:
    print("🚨 Violation Detected: Triple Riding")
else:
    print("✅ No Triple Riding Violation")

# -------------------------------
# Display output image
# -------------------------------
cv2.imshow("Triple Riding Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
