import os
import sys
import cv2
import math
import matplotlib.pyplot as plt
import time

# used for MRCNN (os, sys, math, numpy as np, matplotlib.pyplot as plt)
import random
import skimage.io
import matplotlib

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.coco import coco

# Root directory of the project
ROOT_DIR = os.path.abspath('/home/ivar/Documents/Thesis/clutterbot/')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
# Local path to your trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mrcnn/weights/mask_rcnn_coco.h5')
HAMMER_MODEL_PATH = os.path.join(ROOT_DIR, 'mrcnn/weights/mask_rcnn_hammer_0010.h5')
CUSTOM_MODEL_PATH = os.path.join(ROOT_DIR, 'mrcnn/weights/mask_rcnn_BEST_0003.h5')

class CustomConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2    # Number of classes (including background)
    NUM_CLASSES = 1 + 16  # Background + (Horse and Man)    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

config = CustomConfig()

# class InferenceConfig(coco.CocoConfig):
class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7
    
config = InferenceConfig()

# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#             'bus', 'train', 'truck', 'boat', 'traffic light',
#             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#             'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#             'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#             'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#             'kite', 'baseball bat', 'baseball glove', 'skateboard',
#             'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#             'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#             'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#             'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#             'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#             'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#             'teddy bear', 'hair drier', 'toothbrush']

class_names = ['BG', 'Banana', 'ChipsCan', 'CrackerBox', 'FoamBrick', 'GelatinBox', 'Hammer', 
                'MasterChefCan', 'MediumClamp', 'MustardBottle', 'Pear', 'PottedMeatCan', 'PowerDrill', 
                'Scissors', 'Strawberry', 'TennisBall', 'TomatoSoupCan']

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on the COCO dataset 
# model.load_weights(COCO_MODEL_PATH, by_name=True)
model.load_weights(CUSTOM_MODEL_PATH, by_name=True)

for name in class_names:
    if name == 'BG': continue
    img = skimage.io.imread('data/train/' + name + '63.jpg')
    results = model.detect([img], verbose=1)

    # get dictionary for first prediction
    image_results = results[0]

    #Visualize results
    r = results[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

# img = skimage.io.imread('trained_models/Mask_RCNN/images/sample.jpg')
# img = skimage.io.imread('data/train/GelatinBox12.jpg')
# img = skimage.io.imread('images/0.jpg')
# plt.figure(figsize=(12,10))
# skimage.io.imshow(img)
# print("IMAGE:", img)
# img = load_img('trained_models/Mask_RCNN/images/sample.jpg')
# img = img_to_array(img)

# results = model.detect([img], verbose=1)
# # get dictionary for first prediction
# image_results = results[0]
# #Visualize results
# r = results[0]
# visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

# box, mask, classID, score = image_results['rois'], image_results['masks'], image_results['class_ids'], image_results['scores']

# # show photo with bounding boxes, masks, class labels and scores
# fig_images, cur_ax = plt.subplots(figsize=(15, 15))
# display_instances(img, box, mask, classID, class_names, score, ax=cur_ax)

# plt.imshow(img)