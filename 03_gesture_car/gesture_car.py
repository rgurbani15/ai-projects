import cv2
import numpy as np
from ultralytics import YOLO

# Load your custom hand detector
hand_model = YOLO('hand_detector.pt')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Car state
car_x, car_y = 320, 400
car_speed = 0
car_angle = 0
target_speed = 0
target_angle = 0
smooth = 0.15

def draw_car(frame, x, y, angle_deg, speed):
    """Draw a race car shape"""
    angle_rad = np.radians(angle_deg)
    length = 40
    width = 20
    
    fx = int(x + length * np.sin(angle_rad))
    fy = int(y - length * np.cos(angle_rad))
    
    blx = int(x - length*0.5 * np.sin(angle_rad) + width * np.cos(angle_rad))
    bly = int(y + length*0.5 * np.cos(angle_rad) + width * np.sin(angle_rad))
    
    brx = int(x - length*0.5 * np.sin(angle_rad) - width * np.cos(angle_rad))
    bry = int(y + length*0.5 * np.cos(angle_rad) - width * np.sin(angle_rad))
    
    pts = np.array([[fx, fy], [blx, bly], [brx, bry]])
    cv2.fillPoly(frame, [pts], (0, 0, 255))
    
    # Wheels
    wheel_offset = 15
    wl_x = int(blx + wheel_offset * np.cos(angle_rad))
    wl_y = int(bly + wheel_offset * np.sin(angle_rad))
    wr_x = int(brx + wheel_offset * np.cos(angle_rad))
    wr_y = int(bry + wheel_offset * np.sin(angle_rad))
    
    cv2.circle(frame, (wl_x, wl_y), 6, (50, 50, 50), -1)
    cv2.circle(frame, (wr_x, wr_y), 6, (50, 50, 50), -1)
    
    # Flame if fast
    if speed > 0.3:
        flame_x = int(x - length*0.8 * np.sin(angle_rad))
        flame_y = int(y + length*0.8 * np.cos(angle_rad))
        cv2.circle(frame, (flame_x, flame_y), int(speed*10), (0, 140, 255), -1)

print("AI Hand-Controlled F1 Car")
print("Show your hand to steer and accelerate")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Detect hands with your custom AI model
    results = hand_model(frame, verbose=False)
    
    hand_pos = None
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        # Get largest hand
        boxes = results[0].boxes
        largest = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
        
        x1, y1, x2, y2 = largest.xyxy[0].cpu().numpy()
        hx = int((x1 + x2) / 2)
        hy = int((y1 + y2) / 2)
        hand_pos = (hx, hy)
        
        # Draw hand box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (hx, hy), 8, (0, 255, 0), -1)
        
        # Map hand position to car controls
        target_angle = (hx - w//2) / (w//2)  # -1 (left) to 1 (right)
        target_speed = -(hy - h//2) / (h//2)  # -1 (bottom) to 1 (top)
        
        target_angle = max(-1, min(1, target_angle))
        target_speed = max(-1, min(1, target_speed))
    else:
        # No hand = coast to stop
        target_speed = 0
        target_angle = 0
    
    # Smooth movement
    car_speed += (target_speed - car_speed) * smooth
    car_angle += (target_angle - car_angle) * smooth
    
    # Update car position
    speed_px = car_speed * 6
    angle_deg = car_angle * 45
    
    car_x += int(np.sin(np.radians(angle_deg)) * speed_px)
    car_y -= int(np.cos(np.radians(angle_deg)) * speed_px)
    
    # Keep car on screen
    car_x = max(30, min(w-30, car_x))
    car_y = max(30, min(h-30, car_y))
    
    # Draw car
    draw_car(frame, car_x, car_y, angle_deg, car_speed)
    
    # HUD
    cv2.putText(frame, f"Speed: {car_speed:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Steer: {angle_deg:.1f}deg", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Hand = joystick | Center = stop | Move to steer/accel", 
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Crosshair
    cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (100, 100, 100), 1)
    cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (100, 100, 100), 1)
    
    cv2.imshow('AI Hand-Controlled F1 Car', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()