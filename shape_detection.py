import cv2
import numpy as np

def detect_shapes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Ignore small contours
            # Approximate the contour
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Classify shape based on number of vertices
            vertices = len(approx)
            
            if vertices == 3:
                shape_name = "Triangle"
                color = (0, 255, 0)  # Green
            elif vertices == 4:
                # Check if square or rectangle
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 0.95 <= aspect_ratio <= 1.05:
                    shape_name = "Square"
                    color = (0, 0, 255)  # Red
                else:
                    shape_name = "Rectangle"
                    color = (255, 0, 0)  # Blue
            elif vertices == 5:
                shape_name = "Pentagon"
                color = (255, 255, 0)  # Cyan
            elif vertices >= 6:
                shape_name = "Circle"
                color = (0, 255, 255)  # Yellow
            else:
                shape_name = "Unknown"
                color = (255, 255, 255)  # White
            
            x, y, w, h = cv2.boundingRect(contour)
            shapes.append({'name': shape_name, 'bbox': (x, y, w, h), 'color': color})
    
    return shapes

print("🔷 Shape Detection Started!")
print("📷 Camera activated")
print("🎯 Detecting: Circles, Squares, Triangles, Rectangles")
print("⏹️  Press 'q' to quit")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    shapes = detect_shapes(frame)
    
    for shape in shapes:
        x, y, w, h = shape['bbox']
        color = shape['color']
        name = shape['name']
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.putText(frame, f'Shapes detected: {len(shapes)}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Shape Detection - Press Q to quit', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Shape detection stopped!")