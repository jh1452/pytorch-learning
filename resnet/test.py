import cv2
import cv2

box_image = image.copy()

cv2.rectangle(
    box_image,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    2
)