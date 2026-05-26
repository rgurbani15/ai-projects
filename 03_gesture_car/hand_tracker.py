import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Car state
car_x, car_y = 320, 400
car_speed = 0
car_angle = 0
target_speed = 0
target_angle = 0

# Smoothing factor (lower = smoother, higher = more responsive)
smooth = 0.15

def detect_hand_center(frame):
    """Find hand using skin color detection. Returns (x, y) or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Skin color range in HSV
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, mask
    
    # Largest contour = hand
    largest = max(contours, key=cv2.contourArea)
    
    # Ignore small blobs (noise)
    if cv2.contourArea(largest) < 5000:
        return None, mask
    
    # Get center
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, mask
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy), mask

print("Hand-Controlled Car")
print("Move hand LEFT/RIGHT to steer, UP/DOWN for speed")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Detect hand
    hand_pos, mask = detect_hand_center(frame)
    
    if hand_pos:
        hx, hy = hand_pos
        
        # Draw hand tracking
        cv2.circle(frame, (hx, hy), 10, (0, 255, 0), -1)
        cv2.circle(frame, (hx, hy), 15, (0, 255, 0), 2)
        
        # Map to control values (-1 to 1)
        # X: left = -1, right = 1
        # Y: top = 1 (fast), bottom = -1 (reverse)
        target_angle = (hx - w//2) / (w//2)  # -1 to 1
        target_speed = -(hy - h//2) / (h//2)  # -1 to 1 (flip Y)
        
        # Clamp
        target_angle = max(-1, min(1, target_angle))
        target_speed = max(-1, min(1, target_speed))
        
        # Draw target marker
        cv2.line(frame, (w//2, h//2), (hx, hy), (0, 255, 0), 2)
    else:
        # No hand = coast to stop
        target_speed = 0
        target_angle = 0
    
    # Smooth movement
    car_speed += (target_speed - car_speed) * smooth
    car_angle += (target_angle - car_angle) * smooth
    
    # Update car position
    car_speed_px = car_speed * 5  # Max 5 pixels per frame
    car_angle_deg = car_angle * 30  # Max 30 degrees
    
    car_x += int(np.sin(np.radians(car_angle_deg)) * car_speed_px)
    car_y -= int(np.cos(np.radians(car_angle_deg)) * car_speed_px)
    
    # Keep car on screen
    car_x = max(20, min(w-20, car_x))
    car_y = max(20, min(h-20, car_y))
    
    # Draw car (simple triangle)
    angle_rad = np.radians(car_angle_deg)
    p1 = (int(car_x + 20 * np.sin(angle_rad)), 
          int(car_y - 20 * np.cos(angle_rad)))
    p2 = (int(car_x + 10 * np.sin(angle_rad + 2.5)), 
          int(car_y - 10 * np.cos(angle_rad + 2.5)))
    p3 = (int(car_x + 10 * np.sin(angle_rad - 2.5)), 
          int(car_y - 10 * np.cos(angle_rad - 2.5)))
    
    cv2.fillPoly(frame, [np.array([p1, p2, p3])], (0, 140, 255))
    
    # Draw HUD
    cv2.putText(frame, f"Speed: {car_speed:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Angle: {car_angle_deg:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Center = Stop | Left/Right = Steer | Up/Down = Speed", 
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw center crosshair
    cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (100, 100, 100), 1)
    cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (100, 100, 100), 1)
    
    cv2.imshow('Hand-Controlled Car', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
