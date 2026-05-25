from ultralytics import YOLO
import cv2

print("Loading model...")
model = YOLO('yolov8n.pt')
print("Model loaded")

print("Opening webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(f"Webcam opened: {cap.isOpened()}")

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

print("Starting loop...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    print("Running detection...")
    results = model(frame)
    annotated = results[0].plot()
    
    cv2.imshow('AI Webcam', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit pressed")
        break

cap.release()
cv2.destroyAllWindows()
print("Done")