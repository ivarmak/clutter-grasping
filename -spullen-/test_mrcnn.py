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

sys.path.append('trained_models/Mask_RCNN')
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
from mrcnn.model import log

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img

from samples.coco import coco

# Root directory of the project
ROOT_DIR = os.path.abspath('/home/ivar/Documents/Thesis/clutterbot/') ## PATH_TO_YOUR_WORK_DIRECTORY
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
# Local path to your trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mrcnn/weights/mask_rcnn_coco.h5')

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
config = InferenceConfig()

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on the COCO dataset 
model.load_weights(COCO_MODEL_PATH, by_name=True)

img = skimage.io.imread('trained_models/Mask_RCNN/images/sample.jpg')
# plt.figure(figsize=(12,10))
# skimage.io.imshow(img)
# print("IMAGE:", img)
# img = load_img('trained_models/Mask_RCNN/images/sample.jpg')
# img = img_to_array(img)

results = model.detect([img], verbose=1)

# get dictionary for first prediction
image_results = results[0]

#Visualize results
r = results[0]
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

# box, mask, classID, score = image_results['rois'], image_results['masks'], image_results['class_ids'], image_results['scores']

# # show photo with bounding boxes, masks, class labels and scores
# fig_images, cur_ax = plt.subplots(figsize=(15, 15))
# display_instances(img, box, mask, classID, class_names, score, ax=cur_ax)

# plt.imshow(img)