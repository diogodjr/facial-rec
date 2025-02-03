import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    results = model.predict(frame, conf=0.6)

    object_count = {}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            label = model.names[int(box.cls[0])]    
            confidence = float(box.conf[0])        

            if confidence > 0.5:
                object_count[label] = object_count.get(label, 0) + 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    y_offset = 30
    for obj, count in object_count.items():
        cv2.putText(frame, f"{obj}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    cv2.imshow("Detecção de Objetos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
