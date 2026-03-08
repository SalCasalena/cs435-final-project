from ultralytics import YOLO
import cv2

#This Model only detects stop signs and traffic lights. On the COCO datasets
model = YOLO("yolo11n.pt")

img = cv2.imread("trafficStop.jpg")

results = model(img)

for r in results:
    if len(r.boxes) == 0:
        print("No objects detected in the image.")
    else:
        print(f"\nFound {len(r.boxes)} object(s):\n")
        for i, box in enumerate(r.boxes, 1):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            
            # Get position info
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            
            print(f"{i}. {class_name.upper()}")
            print(f"   - Confidence: {confidence:.1%} (The model is {confidence:.1%} sure this is a {class_name})")
            print(f"   - Size: {width:.0f}x{height:.0f} pixels (The object takes up this much space)")
            print(f"   - Location: Center at ({(x1+x2)/2:.0f}, {(y1+y2)/2:.0f}) in the image")
            print()

results[0].show()