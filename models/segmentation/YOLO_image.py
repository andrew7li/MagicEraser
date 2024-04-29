import sys

import cv2
import numpy as np
from ultralytics import YOLO

# NOTE: CHANGE ME
FILE_NAME = 'bus-students.jpg'
img = cv2.imread("../../data/" + FILE_NAME)

# Hyperparameter
CONF_THRESHOLD = 0.5

def detect_objects(model):
    """
    Given the model, detect and print all objects where confidence exceeds CONF_THRESHOLD.
    """
    # Detect objects
    results = model.predict(img, conf=CONF_THRESHOLD)

    # Loop over detected objects and print them
    for result in results:
        for idx, (_, box) in enumerate(zip(result.masks.xy, result.boxes)):
            class_id = int(box.cls)
            object_name = model.names[class_id]
            print(f"ID: {idx} of Object: {object_name}")
        break
    return results

if __name__ == "__main__":
    arguments = sys.argv
    if not(len(arguments) == 1 or len(arguments) == 2):
        raise ValueError("Arguments must be of length 1 or 2! Please try again!")

    # Load the model and detect objects
    model = YOLO("yolov8m-seg.pt")
    results = detect_objects(model)

    if len(arguments) == 1:
        for idx, (mask, box) in enumerate(zip(results[0].masks.xy, results[0].boxes)):
            points = np.int32([mask])
            cv2.fillPoly(img, points, color=(0, 0, 0))
    else:
        index = int(arguments[1])
        mask1 = results[0].masks.xy[index]
        points = np.int32([mask1])
        cv2.fillPoly(img, points, color=(0, 0, 0))
        
    # Display image
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    # Save image
    cv2.imwrite("../../results/" + FILE_NAME, img)
