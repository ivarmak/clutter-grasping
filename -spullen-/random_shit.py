img = bgr
dimensions = img.shape
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
print('Image Dimension    : ',dimensions)
print('Image Height       : ',height)
print('Image Width        : ',width)
print('Number of Channels : ',channels)

## zoek in dubbele array (mask) wat niet achtergrond is en print dat
# for j in seg:
#     for i in filter(lambda x: x != 4 and x != 1, j):
#         print(i)
# cv2.imwrite('MRCNNoutput/banaan.jpg',rgb)

# plt.figure(figsize=(12,10))
# skimage.io.imshow(img)
# print("IMAGE:", img)
# img = load_img('trained_models/Mask_RCNN/images/sample.jpg')
# img = img_to_array(img)

# results = model.detect([img], verbose=1)

# get dictionary for first prediction
# image_results = results[0]

#Visualize results
# r = results[0]
# visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

# box, mask, classID, score = r['rois'], r['masks'], r['class_ids'], r['scores']

# # show photo with bounding boxes, masks, class labels and scores
# fig_images, cur_ax = plt.subplots(figsize=(15, 15))
# display_instances(img, box, mask, classID, class_names, score, ax=cur_ax)

# plt.imshow(img)

# def banana_scenarioOLD(self, device, vis, output, debug):
# objects = YcbObjects('objects/ycb_objects',
#                     mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
#                     mod_stiffness=['Strawberry'])

# banana_path = 'objects/ycb_objects/YcbBanana/model.urdf'

# ## camera settings: cam_pos, cam_target, near, far, size, fov
# center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
# # camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)

# MRCNN_IMG_SIZE = 1024
# camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (MRCNN_IMG_SIZE, MRCNN_IMG_SIZE), 40)
# env = Environment(camera, vis=vis, debug=debug, finger_length=0.06)

# env.reset_robot()          
# env.remove_all_obj()                        

# # load banana into environment
# env.load_isolated_obj(banana_path)
# self.dummy_simulation_steps(20)

# bgr, depth, _ = camera.get_cam_img()
# ##convert BGR to RGB
# # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # print('')
        # print('box: ', box)
        # print('mask: ', mask)
        # print('classID', classID)
        # print('score', score)
        # print('')

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

config = InferenceConfig()
# config.display()

# COCO Class names
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
model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

# Load weights trained on MS-COCO
model.load_weights('trained_models/Mask_RCNN/samples/mask_rcnn_coco.h5', by_name=True)

## Strange, BGR is actually rgb colours
# plt.imshow(bgr)
# plt.show()

# Load a random image from the images folder
image = skimage.io.imread('trained_models/Mask_RCNN/images/sample.jpg')

# original image
plt.figure(figsize=(12,10))
skimage.io.imshow(image)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
