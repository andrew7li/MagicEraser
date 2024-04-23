import cv2
from ultralytics import YOLO
import time

model = YOLO('yolov8n-seg.pt')
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
color = (255, 0, 0)  # Blue color in BGR
thickness = 2
line_type = cv2.LINE_AA


while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        break
    
    start = time.perf_counter()
    results = model(frame)

    end = time.perf_counter()
    total_time = end - start
    fps = 1 / total_time

    annotated_frame = results[0].plot()

    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), font, font_scale, color, thickness, line_type)
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()