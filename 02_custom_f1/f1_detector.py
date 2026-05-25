from ultralytics import YOLO
import cv2

# Load your custom trained model
model = YOLO('best.pt')

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection - only finds f1_car class
    results = model(frame)
    annotated = results[0].plot()
    
    cv2.imshow('F1 Car Detector', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()