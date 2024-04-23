from ultralytics import YOLO
import random
import cv2
import numpy as np

# NOTE: CHANGE ME
FILE_NAME = 'bus-students.jpg'

# Hyperparameter
CONF_THRESHOLD = 0.5

# Load the model
model = YOLO("yolov8m-seg.pt")
img = cv2.imread("../../data/" + FILE_NAME)

# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

results = model.predict(img, conf=CONF_THRESHOLD)
colors = [random.choices(range(256), k=3) for _ in classes_ids]

print(len(results))
for result in results:
    for idx, (mask, box) in enumerate(zip(result.masks.xy, result.boxes)):
        points = np.int32([mask])
        # cv2.polylines(img, points, True, (255, 0, 0), 1)
        color_number = classes_ids.index(int(box.cls[0]))
        # cv2.fillPoly(img, points, colors[color_number])
        # cv2.fillPoly(img, points, color=(0, 0, 0))

        class_id = int(box.cls)
        object_name = model.names[class_id]
        print(f"ID: {idx} of Object: {object_name}")
    break

index = 5
mask1 = results[0].masks.xy[index]
points = np.int32([mask1])
cv2.fillPoly(img, points, color=(0, 0, 0))

# for result in results:
#     for idx, (mask, box) in enumerate(zip(obj.masks.xy, obj.boxes)):
#         points = np.int32([mask])
#         # cv2.polylines(img, points, True, (255, 0, 0), 1)
#         color_number = classes_ids.index(int(box.cls[0]))
#         # cv2.fillPoly(img, points, colors[color_number])
#         cv2.fillPoly(img, points, color=(0, 0, 0))
#         class_id = int(box.cls)
#         object_name = model.names[class_id]
#         print(f"ID: {idx} of Object: {object_name}")





cv2.imshow("Image", img)
cv2.waitKey(0)

cv2.imwrite("../../results/" + FILE_NAME, img)
