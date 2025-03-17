from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')  # Medium-sized YOLOv8

# Perform inference on the image with a lower confidence threshold
results = model('./play03/frame_001.jpg', conf=0.1)

for result in results:
    # Get a copy of the original image
    img = result.orig_img.copy()

    # Get bounding boxes, class IDs, and confidence scores from the results
    boxes = result.boxes.xyxy.cpu().numpy()     # each box: [x1, y1, x2, y2]
    classes = result.boxes.cls.cpu().numpy()      # class indices
    confidences = result.boxes.conf.cpu().numpy()   # confidence scores

    # Filter to keep only the "person" detections (COCO 'person' class is 0)
    person_boxes = []
    person_confidences = []
    for box, cls, conf in zip(boxes, classes, confidences):
        if int(cls) == 0:
            person_boxes.append(box)
            person_confidences.append(float(conf))

    # Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] for OpenCV's NMS function
    boxes_xywh = []
    for box in person_boxes:
        x1, y1, x2, y2 = box.astype(int)
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

    # Apply Non-Maximum Suppression to combine overlapping boxes.
    # Adjust the NMS threshold (e.g., 0.4) as needed.
    indices = cv2.dnn.NMSBoxes(boxes_xywh, person_confidences, score_threshold=0.2, nms_threshold=0.4)

    # Draw only the merged bounding boxes on the image.
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes_xywh[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"person {person_confidences[i]:.2f}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Combined Person Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
