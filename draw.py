import random

import cv2
from matplotlib import pyplot as plt
import albumentations as A

def visualize_bbox(img, bbox, box_color=(255, 0, 0), box_thickness=2):
    x_min, y_min, x_max, y_max = list(map(int,bbox))
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=box_color, thickness=box_thickness)
    
    return img

def visualize(image, bboxes):
    img = image.copy()
    for bbox in bboxes:
        img = visualize_bbox(img, bbox)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)

def save_img(image, bboxes, path, name):
    img = image.copy()
    for bbox in bboxes:
        img = visualize_bbox(img, bbox)

    img_name = path + name + '.jpg'
    cv2.imwrite(img_name, img)
