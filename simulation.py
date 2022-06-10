from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.env import Environment
from utils import YcbObjects, PackPileData, IsolatedObjData, summarize
import numpy as np
import pybullet as p
import argparse
import os
import sys
import json
import cv2
import math
import matplotlib.pyplot as plt
import time
import skimage.io
import random
from PIL import Image

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.coco import coco

# Root directory of the project
ROOT_DIR = os.path.abspath('/home/ivar/Documents/Thesis/clutterbot/')

def setup_mrcnn(weights, weights_name, conf = 0.9):
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
    # Local path to your trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mrcnn/weights/mask_rcnn_coco.h5')
    CUSTOM_MODEL_PATH = os.path.join(ROOT_DIR, 'mrcnn/weights/' + weights_name)

    if weights == 'coco':
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
        weights_path = COCO_MODEL_PATH

    elif weights == 'custom':
        class CustomConfig(Config):
            NAME = "object"
            IMAGES_PER_GPU = 1              # Adjust down if you use a smaller GPU.
            NUM_CLASSES = 1 + 16             # Background + Hammer   
            STEPS_PER_EPOCH = 100           # Number of training steps per epoch
            DETECTION_MIN_CONFIDENCE = 0.5  # Skip detections with < 90% confidence

        config = CustomConfig()

        class InferenceConfig(config.__class__):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = conf
            
        config = InferenceConfig()

        class_names = ['BG', 'Banana', 'ChipsCan', 'CrackerBox', 'FoamBrick', 'GelatinBox', 'Hammer', 
                'MasterChefCan', 'MediumClamp', 'MustardBottle', 'Pear', 'PottedMeatCan', 'PowerDrill', 
                'Scissors', 'Strawberry', 'TennisBall', 'TomatoSoupCan']

        weights_path = CUSTOM_MODEL_PATH

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(weights_path, by_name=True)

    return model, class_names

def evaluate_mrcnn(model, rgb):
    print("\nRecognition phase...")
    start = time.time()

    results = model.detect([rgb], verbose=0)
    r = results[0]
    box, mask, classID, score = r['rois'], r['masks'], r['class_ids'], r['scores']                      

    end = time.time()
    print('MRCNN execution time: ', end - start)

    return box, mask, classID, score

def look_at_object(vis):
    CAM_Z = 1.9
    IMG_SIZE = 224
    MRCNN_IMG_SIZE = 448

    weights = 'bestMRCNN_1000st_20ep_augSeg_gt1_val0.19'
    weights2 = 'MRCNN_st300_20ep_augSeq_GT1_val0.18'
    weights3 = 'mask_rcnn_object_0032'
    weights4 = 'rand/rand_4000st/weights.bestVal=0.22.hdf5'
    weights5 = 'tex/tex100_800st2_endEp30_val0.24/weights.bestVal.hdf5'

    model, class_names = setup_mrcnn('custom', weights5, 0.8)

    objects = YcbObjects('objects/ycb_objects',
                        mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                        mod_stiffness=['Strawberry'])

    names = ['BG', 'Banana', 'ChipsCan', 'CrackerBox', 'FoamBrick', 'GelatinBox', 'Hammer', 
                'MasterChefCan', 'MediumClamp', 'MustardBottle', 'Pear', 'PottedMeatCan', 'PowerDrill', 
                'Scissors', 'Strawberry', 'TennisBall', 'TomatoSoupCan']

    obj = 'Banana'

    obj_path = 'objects/ycb_objects/Ycb' + obj + '/model.urdf'

    ## camera settings: cam_pos, cam_target, near, far, size, fov
    center_x, center_y, center_z = 0.05, -0.52, CAM_Z

    # camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (IMG_SIZE, IMG_SIZE), 40)
    camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (MRCNN_IMG_SIZE, MRCNN_IMG_SIZE), 40)
    env = Environment(camera, vis=True, finger_length=0.06)

    env.reset_robot()          
    env.remove_all_obj()                        
    
    # load object into environment
    # env.load_isolated_obj(obj_path)

    ## load turned object
    pitch = bool(random.getrandbits(1))
    roll = bool(random.getrandbits(1))
    env.load_turnable_obj(obj_path, pitch, roll)

    # load pile of objects
    number_of_objects = 5
    objects.shuffle_objects()
    info = objects.get_n_first_obj_info(number_of_objects)

    # env.create_packed(info)
    env.create_pile(info)

    rgb, _, seg = camera.get_cam_img()

    box, mask, classID, score = evaluate_mrcnn(model, rgb)
    # print(classID)
    # plt.imshow(seg)

    visualize.display_instances(rgb, box, mask, classID, class_names, score)
    def transform_coordinates(box, img_size=448):
        XMIN, XMAX, YMIN, YMAX = [-0.35, 0.45, -0.92, -0.12]

        y1, x1, y2, x2 = box

        y1 = (((y1 / img_size) * 0.8) + abs(YMAX))*-1
        y2 = (((y2 / img_size) * 0.8) + abs(YMAX))*-1
        x1 = ((x1 / img_size) * 0.8) - abs(XMIN)
        x2 = ((x2 / img_size) * 0.8) - abs(XMIN)

        return [y1,x1,y2,x2]
    
    recogObjects = []

    def object_is_isolated(box, recogObjects):
        y1, x1, y2, x2 = box

        for i in range(y2-y1):
            for j in range(x2-x1):
                for obj in recogObjects:
                    if obj["mask"][j,i]:
                        return False

        return True

    box = box[0]

    def isolate_object(box):

        padding = 17
        box[0] -= padding
        box[1] -= padding
        box[2] += padding
        box[3] += padding

        box = transform_coordinates(box)

        y1,x1,y2,x2 = box
        vert = p.getQuaternionFromEuler([0.5*np.pi, np.pi/2, 0.0])
        hor = p.getQuaternionFromEuler([0.0, np.pi/2, 0.0])
        y_orn = [hor, vert, hor, vert, hor]

        x = [x1, x2, x2, x1, x1]
        y = [y1, y1, y2, y2, y1]

        ## TODO: make dependent on depth image
        c_idx = 0

        env.move_gripper(0.1)
        env.auto_close_gripper()

        for i in range(5):
            env.move_ee([x[c_idx], y[c_idx], z, y_orn[c_idx]])
            env.move_ee([x[c_idx], y[c_idx], z, y_orn[c_idx+1]])
            
            if i != 4: c_idx = (c_idx+1) %4
        env.move_ee([x[i], y[i], env.GRIPPER_MOVING_HEIGHT, y_orn])

    # isolate_object(box)

    def other_objects_grasped(grasp, obj, recogObjects):
        x,y,z, yaw, opening_len, obj_height = grasp
        gripper_size = opening_len + 0.02
        x1 = x+gripper_size*math.sin(yaw)
        x2 = x-gripper_size*math.sin(yaw)
        y1 = y+gripper_size*math.cos(yaw)
        y2 = y-gripper_size*math.cos(yaw)

        g = self.transform_meters([y1,x1,y2,x2])
        print("g (y1,x1,y2,x2): ", g)


def look_at_banana(vis):
    CAM_Z = 1.9
    IMG_SIZE = 224

    model, class_names = setup_mrcnn('custom', 'tex/tex100_800st2_endEp30_val0.24/weights.bestVal.hdf5')

    objects = YcbObjects('objects/ycb_objects',
                        mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                        mod_stiffness=['Strawberry'])
    
    banana_path = 'objects/ycb_objects/YcbBanana/model.urdf'

    ## camera settings: cam_pos, cam_target, near, far, size, fov
    center_x, center_y, center_z = 0.05, -0.52, CAM_Z
    # camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)
    
    MRCNN_IMG_SIZE = 448
    camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (MRCNN_IMG_SIZE, MRCNN_IMG_SIZE), 40)
    env = Environment(camera, vis=vis, finger_length=0.06)
    
    # env.reset_robot()          
    # env.remove_all_obj()                        
    
    # # load banana into environment
    # env.load_isolated_obj(banana_path)

    # number_of_objects = 5
    # # objects.shuffle_objects()
    # # info = objects.get_n_first_obj_info(number_of_objects)
    # # env.create_pile(info)

    # rgb, _, _ = camera.get_cam_img()

    # bananaFound = False
    # nfNumber = 1

    # box, mask, classID, score = evaluate_mrcnn(model, rgb)

    while(True):
        env.reset_robot()          
        env.remove_all_obj()
        env.load_isolated_obj(banana_path)

        rgb, _, _ = camera.get_cam_img()   
        box, mask, classID, score = evaluate_mrcnn(model, rgb)
        print(box)
        visualize.display_instances(rgb, box, mask, classID, class_names, score)
                     
    
    # load banana into environment
    # env.load_isolated_obj(banana_path)

    # while(not bananaFound):
    #     if (47 in classID):
    #         print('BANANA FOUND')
    #         bananaFound = True

    #         result = np.where(classID == 47)
    #         index = result[0][0]
    #         print('index: ', index)
    #         # terminaloutput>> index:  (array([3]),)
    #         # pak eerste output: result = np.where(classID == 47) \ index = result[0][0]
    #         # https://thispointer.com/find-the-index-of-a-value-in-numpy-array/
    #     else:
    #         print('NOT FOUND, starting again')

    #         # visualize.display_instances(rgb, box, mask, classID, class_names, score, title='notfound ' + str(nfNumber))
            
    #         nfNumber += 1

    #         env.reset_robot()          
    #         env.remove_all_obj()
    #         objects.shuffle_objects()
    #         info = objects.get_n_first_obj_info(number_of_objects)
    #         env.create_pile(info)

    #         rgb, _, _ = camera.get_cam_img()

    #         box, mask, classID, score = evaluate_mrcnn(model, rgb)
    
    # print('NOT FOUND #', nfNumber)
    visualize.display_instances(rgb, box, mask, classID, class_names, score)

def make_data(colab, background):
    CAM_Z = 1.9
    IMG_SIZE = 448

    ## camera settings: cam_pos, cam_target, near, far, size, fov
    center_x, center_y, center_z = 0.05, -0.52, CAM_Z
    camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (IMG_SIZE, IMG_SIZE), 40)
    env = Environment(camera, vis=False, finger_length=0.06)

    train_or_val = 'val'
    nr_of_objects = 75

    object_names = ['Banana', 'ChipsCan', 'CrackerBox', 'FoamBrick', 'GelatinBox', 'Hammer',
                'MasterChefCan', 'MediumClamp', 'MustardBottle', 'Pear', 'PottedMeatCan', 'PowerDrill', 
                'Scissors', 'Strawberry', 'TennisBall', 'TomatoSoupCan']

    dict = {}
    texture_path = 'images/textures/use/'
    save_dir = 'data/' + train_or_val + '/'
    width, height = IMG_SIZE, IMG_SIZE

    def no_object_found():
        _ , depth, _ = camera.get_cam_img()
        if (depth.max()- depth.min() < 0.0025):
            return True
        else:
            return False  

    for obj_name in object_names:
        print(obj_name)
        obj_path = 'objects/ycb_objects/Ycb{}/model.urdf'.format(obj_name)
        id = object_names.index(obj_name) + 1

        ## loop for number of object instances
        for obj_nr in range(nr_of_objects):
        
            env.reset_robot()          
            env.remove_all_obj()

            ## make objects turn randomly
            pitch = bool(random.getrandbits(1))
            roll = bool(random.getrandbits(1))
            env.load_turnable_obj(obj_path, pitch, roll)

            ## fix for scissors that bounce of the table
            if(obj_name == 'Scissors'):
                for _ in range(20):
                        p.stepSimulation()
                if(no_object_found()):
                    env.remove_all_obj()
                    env.load_turnable_obj(obj_path, pitch, roll)
                
            rgb, _, seg = camera.get_cam_img()


            img_name = obj_name + str(obj_nr)
            img_path = save_dir + img_name + '.jpg'
            if (colab): load_path = 'drive/MyDrive/scriptie/' + img_path
            else: load_path = img_path

            ## use np filter for finding mask coordinates (mask value is int 6)
            mask_coord = np.where(seg == 6)

            inst = {
                "name": img_name,
                "path": load_path,
                "obj_id": id,
                "width": width,
                "height": height,
                "mask_x": mask_coord[1].tolist(),
                "mask_y":  mask_coord[0].tolist()        
            }

            dict[img_name] = inst

            if(background=='jitter'):
                ## Initialize randomized RGB value, paste masked image, save
                imarray = np.random.rand(448,448,3) * 255
                imarray[mask_coord[0],mask_coord[1]] = rgb[mask_coord[0],mask_coord[1]]
                im = Image.fromarray(imarray.astype('uint8'))
                im.save(img_path)

            elif(background=='texture'):
                ## Add random texture as background, paste masked image, save
                im_name = random.choice(os.listdir('images/textures/use'))
                texture = Image.open(texture_path + im_name)
                resized = np.asarray(texture.resize((448,448)))
                resized[mask_coord[0],mask_coord[1]] = rgb[mask_coord[0],mask_coord[1]]
                im = Image.fromarray(resized.astype('uint8'))
                im.save(img_path)

            else:
                ## Standard image saving
                plt.imsave(img_path, rgb)
            
    json_path = save_dir + '/' + train_or_val + '_img_data.json'
    with open(json_path, "w") as write:
        json.dump(dict, write)

class GrasppingScenarios():

    def __init__(self,network_model="GR_ConvNet"):
        
        self.network_model = network_model

        if (network_model == "GR_ConvNet"):
            ##### GR-ConvNet #####
            self.IMG_SIZE = 224
            self.network_path = 'trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
            sys.path.append('trained_models/GR_ConvNet')
        elif (network_model == "GG_CNN"):
            self.IMG_SIZE = 300
            self.network_path = 'trained_models/GG_CNN/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'
            sys.path.append('trained_models/GG_CNN')
        elif (network_model == "GG2"):
            self.IMG_SIZE = 300
            self.network_path = 'trained_models/GG_CNN/ggcnn2_weights_cornell/epoch_50_cornell'
            sys.path.append('trained_models/GG_CNN')
        else:
            print("The selected network has not been implemented yet!")
            exit() 
        
        
        self.CAM_Z = 1.9
        self.depth_radius = 1
        self.ATTEMPTS = 5
        self.fig = plt.figure(figsize=(10, 10))
        self.state = "idle"
        self.grasp_idx = 0
       
                
    def draw_predicted_grasp(self,grasps,color = [0,0,1],lineIDs = []):
        x, y, z, yaw, opening_len, obj_height = grasps

        gripper_size = opening_len + 0.02 
        finger_size = 0.075
        # lineIDs = []
        lineIDs.append(p.addUserDebugLine([x, y, z], [x, y, z+0.15],color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        lineIDs.append(p.addUserDebugLine([x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        
        return lineIDs
    
    def remove_drawing(self,lineIDs):
        if lineIDs != []:
            for line in lineIDs:
                p.removeUserDebugItem(line)
    
    def transform_coordinates(self, box, img_size=448):
        XMIN, XMAX, YMIN, YMAX = [-0.35, 0.45, -0.92, -0.12]

        y1, x1, y2, x2 = box

        y1 = (((y1 / img_size) * 0.8) + abs(YMAX))*-1
        y2 = (((y2 / img_size) * 0.8) + abs(YMAX))*-1
        x1 = ((x1 / img_size) * 0.8) - abs(XMIN)
        x2 = ((x2 / img_size) * 0.8) - abs(XMIN)

        return [y1,x1,y2,x2]

    def transform_meters(self, box, img_size=448):

        y1,x1,y2,x2 = box

        y1 = int(((abs(y1) - 0.12) /0.8) * img_size)
        y2 = int(((abs(y2) - 0.12) /0.8) * img_size)
        x1 = int(((x1 + 0.35) / 0.8) * img_size)
        x2 = int(((x2 + 0.35) / 0.8) * img_size)

        return [y1,x1,y2,x2]

    def draw_box(self, box, img_size = 448, color = [0,0,1]):
        if (box == []):
            print("Cannot draw EMPTY BOUNDING BOX\n")
            return box
        
        Z = 0.785 # height of workspace thus z variable line
        lines = []

        ## add a buffer zone so bounding boxes that are a little to tight do not decrease grasp performance
        # box = self.add_padding_to_box(2,box)

        newBox = self.transform_coordinates(box, img_size)
        y1, x1, y2, x2 = newBox

        lines.append(p.addUserDebugLine([x1, y1, Z], [x1, y2, Z], color, lineWidth=3))
        lines.append(p.addUserDebugLine([x2, y1, Z], [x2, y2, Z], color, lineWidth=3))
        lines.append(p.addUserDebugLine([x1, y1, Z], [x2, y1, Z], color, lineWidth=3))
        lines.append(p.addUserDebugLine([x1, y2, Z], [x2, y2, Z], color, lineWidth=3))

        self.write_temp_text("Grasping from box: ", [0,0,1])
        return lines

    def write_temp_text(self, text, color = [0,0.5,0]):
        debugID = p.addUserDebugText(text, [-0.0, -1.2, 0.8], color, textSize=2)
        time.sleep(0.8)
        p.removeUserDebugItem(debugID)

    def dummy_simulation_steps(self,n):
        for _ in range(n):
            p.stepSimulation()

    def write_perm_text(self,previoustext, text, color = [0.5,0,0], loc = [-0.15,-0.52,1.92], textSize=2):
        if previoustext != "": p.removeUserDebugItem(previoustext)
        debugID = p.addUserDebugText(text, loc, color, textSize)
        return debugID

    def run_mrcnn(self, model, class_names, rgb, min_conf, target, pile = False):
        print(f"\nRecog phase - conf threshold: {min_conf} - ", end= " ")

        start = time.time()

        recogObjects = {}
        targetIndex = np.NaN
        objectTexts = []

        results = model.detect([rgb], verbose=0)
        r = results[0]
        box, mask, classIDs, score = r['rois'], r['masks'], r['class_ids'], r['scores']

        # visualize.display_instances(rgb, box, mask, classIDs, class_names, score)                      
        end = time.time()
        print('exec time: {:.2f}'.format(end - start))

        for j in range(classIDs.size):
            if score[j] > min_conf:

                found_obj = class_names[classIDs[j]]
                ## skip crackerbox in piled scenario, ungraspable if fallen over
                if pile and found_obj == "CrackerBox":
                    continue
                if self.state != "targetFound":
                    self.change_state("nonTargetFound")
                    if found_obj == target: 
                        self.change_state("targetFound")
                        targetIndex = j
                convBox = self.transform_coordinates(box[j])
                obj = {
                    "id" : classIDs[j],
                    "name": found_obj,
                    "box": box[j],
                    "convBox": convBox,
                    "mask": mask[:,:,j],
                    "score": score[j],
                }
                objectTexts.append(self.write_perm_text("", class_names[classIDs[j]], [0,0,0.5], [convBox[3],convBox[2],0.85], 1))
                recogObjects[j] = obj
            print("{} {:.2f} | ".format(class_names[classIDs[j]], score[j]), end=" ")
        print("")

        return recogObjects, targetIndex, objectTexts

    def is_there_any_object(self,camera):
        self.dummy_simulation_steps(10)
        rgb, depth, _ = camera.get_cam_img()
        #print ("min RGB = ", rgb.min(), "max RGB = ", rgb.max(), "rgb.avg() = ", np.average(rgb))
        #print ("min depth = ", depth.min(), "max depth = ", depth.max())
        if (depth.max()- depth.min() < 0.0025):
            return False
        else:
            return True         

    def spawn_four_objects(self, objects, spawn_obj, env):
        LOCATIONS = [(-0.1, -0.4), (-0.1, -0.6), (0.2, -0.6), (0.2, -0.4)]
        random.shuffle(LOCATIONS)

        for obj in spawn_obj:
            path, mod_orn, mod_stiffness = objects.get_obj_info(obj)
            env.load_obj_same_place(path, LOCATIONS[spawn_obj.index(obj)][0], LOCATIONS[spawn_obj.index(obj)][1], mod_orn, mod_stiffness)

    def change_state(self, newState):
        # print("\nSTATECHANGE: {} -> {}".format(self.state, newState))
        self.state = newState

    def add_padding_to_box(self, padding, box):
        box[0] -= padding
        box[1] -= padding
        box[2] += padding
        box[3] += padding
        return box

    def isolate_object(self, box, env):
        print("Isolating object...")
        box = self.add_padding_to_box(17, box)

        box = self.transform_coordinates(box)

        y1,x1,y2,x2 = box
        vert = p.getQuaternionFromEuler([0.5*np.pi, np.pi/2, 0.0])
        hor = p.getQuaternionFromEuler([0.0, np.pi/2, 0.0])
        y_orn = [hor, vert, hor, vert, hor]

        x = [x2, x1, x1, x2, x2]
        y = [y2, y2, y1, y1, y2]
        Z = 1.025

        ## TODO: make dependent on depth image?
        c_idx = 0

        env.move_gripper(0.1)
        env.auto_close_gripper()

        for i in range(5):
            env.move_ee([x[c_idx], y[c_idx], Z, y_orn[c_idx]])
            env.move_ee([x[c_idx], y[c_idx], Z, y_orn[c_idx+1]])

            if i != 4: c_idx = (c_idx+1) %4
            # input("Press Enter to continue...")
            
        env.move_ee([x[c_idx], y[c_idx], env.GRIPPER_MOVING_HEIGHT, y_orn[c_idx]])
        env.reset_robot()    
    
    def object_is_isolated(self, box, obj_in_box, recogObjects):
        y1, x1, y2, x2 = box

        for i in range(y2-y1):
            for j in range(x2-x1):
                for obj in recogObjects.values():
                    if obj["mask"][y1+i,x1+j] and (obj["name"] != obj_in_box):
                        # print("{} is not {}".format(obj["name"], obj_in_box))
                        print("{} is overlapping at coordinates x:{}, y:{}".format(obj["name"], x1+j, y1+i))
                        return False

        return True
    
    def masks_intersect(self, graspObject, recogObjects):
        for obj in recogObjects.values():
            print("checking intersection for: ", obj["name"])
            if obj["name"] != graspObject["name"]:
                intersect = obj["mask"]*graspObject["mask"]
                if intersect.any():
                    print("Intersection found between masks")
                    return True
                else:
                    print("Masks are completely free of intersections")
        return False


    def other_objects_grasped(self, grasp, graspObject, recogObjects):
        x,y,z, yaw, opening_len, obj_height = grasp
        gripper_size = opening_len + 0.02
        x1 = x+gripper_size*math.sin(yaw)
        x2 = x-gripper_size*math.sin(yaw)
        y1 = y+gripper_size*math.cos(yaw)
        y2 = y-gripper_size*math.cos(yaw)

        y1,x1,y2,x2 = self.transform_meters([y1,x1,y2,x2])
        
        grasp_mask = np.zeros((448,448))

        for i in range(y2-y1):
            for j in range (x2-x1):
                grasp_mask[y1+i,x1+j] = 1
        
        for obj in recogObjects.values():
            if obj["name"] != graspObject["name"]:
                intersect = obj["mask"]*graspObject["mask"]
                if intersect.any():
                    print("Grasp hits mask of: ", obj["name"])
                    return True
        
        print("Grasp is free of other object masks")
        return False

    def isolated_target_scenario(self,runs, device, vis, output, debug):
        model, class_names = setup_mrcnn('custom', 'tex/tex100_800st2_endEp30_val0.24/weights.bestVal.hdf5', 0.4)
        objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'])

        
        ## reporting the results at the end of experiments in the results folder
        data = IsolatedObjData(objects.obj_names, runs, 'results')

        ## camera settings: cam_pos, cam_target, near, far, size, fov
        center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
        camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)
        mrcnn_cam = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (448, 448), 40)
        env = Environment(camera, vis=vis, debug=debug, finger_length=0.06)
        
        generator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)

        """
        Possible states:
        idle
        targetFound
        nonTargetFound
        nothingFound
        targetGrasp
        nonTargetGrasp
        movedToRecogArea
        graspFailed
        """

        targettext = ""
        objects.shuffle_objects()
        target_list = objects.obj_names.copy()
        
        # target_list = ["MasterChefCan"] + target_list

        for target in target_list:
            self.state = "idle"

            for i in range(runs):
                # print("----------- run ", i+1, " -----------")
                # print ("network model = ", self.network_model)
                # print("\nTarget object = ", target)
                if vis: targettext = self.write_perm_text(targettext, "Target: {}".format(target))
                
                ## Shuffle to-be-spawned-objects and remove target so as not to spawn twice
                objects.shuffle_objects()
                other_obj = objects.obj_names.copy()
                other_obj.remove(target)

                env.reset_robot()          
                env.remove_all_obj()                        

                spawn_obj = [target] + other_obj[0:3]
                self.spawn_four_objects(objects, spawn_obj, env)

                # print("Other objects: {}".format(spawn_obj[1:]))

                self.dummy_simulation_steps(20)

                number_of_attempts = self.ATTEMPTS
                number_of_failures = 0

                objectTexts = []
                visualTargetBox = []
                targetDelivered = False

                self.grasp_idx = 0 ## select the best grasp configuration
                failed_to_find_grasp_count = 0
                min_conf = 0.85

                while self.is_there_any_object(camera) and number_of_failures < number_of_attempts and targetDelivered != True:     
                    
                    rgb, depth, _ = camera.get_cam_img()

                    ##########################################################################
                    ## RECOGNITION
                    ##########################################################################

                    mrcnnRGB, _, _ = mrcnn_cam.get_cam_img()

                    # visualize.display_instances(mrcnnRGB, box, mask, classIDs, class_names, score)
                    bbox = []

                    recogObjects, targetIndex, objectTexts = self.run_mrcnn(model, class_names, mrcnnRGB, min_conf, target)

                    ## Target is found on the table, find best grasp point inside bounding box    
                    if (self.state == "targetFound"):
                        self.change_state("targetGrasp")
                        print('\nTARGET {} FOUND'.format(target))
                        targettext = self.write_perm_text(targettext, "{} found".format(target), [0,0.5,0])

                        targetBox = recogObjects[targetIndex]["box"]
                        ## Resize to 224 for GR ConvNet
                        bbox = (targetBox/2).astype(int)
                        if vis: visualTargetBox = self.draw_box(bbox, 224,[0,0.5,0])

                    ## At least one object found on table, grasp non-target object with highest score
                    elif (self.state == "nonTargetFound"):
                        self.change_state("nonTargetGrasp")
                        nonTarget = recogObjects[0]
                        bbox = (nonTarget["box"]/2).astype(int)
                        
                        if vis: 
                            self.write_temp_text("Target not found", [0.5,0,0])
                            self.write_temp_text("Removing {}".format(nonTarget["name"]), [0.5,0,0])
                        print("\nTarget not found, removing {}".format(nonTarget["name"]))
                    
                    ## Object has been moved to recognition area and still not recognized, 
                    ## Lower confidence and restart loop
                    elif (self.state == "movedToRecogArea"):
                        min_conf -= 0.1
                        print("Object has been moved to recog area, and still not recognized.")
                        print("Lowering detection confidence with 0.1 to. ", min_conf)
                        continue
                        
                    ## No object is found on table, freely chosen grasp towards recognition area
                    else:
                        self.change_state("nothingFound")
                        print("\nNo object found on table")
                        if vis: 
                            self.write_temp_text("Object(s) on table, but not recognized", [0.5,0,0])
                            self.write_temp_text("Moving object to recognition area", [0.5,0,0])

                    ##########################################################################
                    ## GRASPING
                    ##########################################################################

                    ## Grasp from bounding box, if empty, grasp is freely chosen
                    grasps, save_name = generator.predict_grasp(rgb, depth, bbox, n_grasps=number_of_attempts, show_output=output)

                    ## NO GRASP POINT FOUND
                    if (grasps == []):
                        self.dummy_simulation_steps(50)
                        if failed_to_find_grasp_count > 3:
                            print("Failed to find a grasp points > 3 times. Skipping.")
                            if vis:
                                self.remove_drawing(lineIDs)
                                self.remove_drawing(objectTexts)
                                self.remove_drawing(visualTargetBox)
                            break
                            
                        failed_to_find_grasp_count += 1                 
                        continue                        

                    ## idx is iterated after incorrect grasp
                    ## check if this next grasp is possible
                    if (self.grasp_idx > len(grasps)-1):  
                        if len(grasps) > 0 :
                            self.grasp_idx = len(grasps)-1
                        else:
                            number_of_failures += 1
                            continue

                    ## DRAWING GRASP
                    if vis:
                        LID =[]
                        for g in grasps:
                            LID = self.draw_predicted_grasp(g,color=[1,0,1],lineIDs=LID)
                        time.sleep(0.5)
                        self.remove_drawing(LID)
                        self.dummy_simulation_steps(10)    
                    lineIDs = self.draw_predicted_grasp(grasps[self.grasp_idx])

                    x, y, z, yaw, opening_len, obj_height = grasps[self.grasp_idx]
                    # print("Performing final grasp, state: {}".format(self.state))
                    
                    ## Target found, move item to target tray
                    if self.state == "targetGrasp":
                        succes_grasp, succes_target, succes_object = env.targeted_grasp((x, y, z), yaw, opening_len, obj_height, target)
                        print("Succesfully grasped target object == {}".format(succes_object))
                        if succes_target:
                            self.write_temp_text("Target dropped successfully")
                            # self.change_state("targetDelivered")
                            targetDelivered = True
                        else:
                            targettext = self.write_perm_text(targettext, "Target: {}".format(target))

                    ## Non-target object recognized, move item to red tray
                    elif self.state == "nonTargetGrasp":
                        succes_grasp, succes_target, succes_object = env.non_target_grasp((x, y, z), yaw, opening_len, obj_height, target)
                        print("Succesfully grasped nontarget object == {}".format(succes_object))

                    ## No object recognized, move to analysis area
                    elif self.state == "nothingFound":
                        succes_grasp, succes_target = env.move_to_recog_area((x, y, z), yaw, opening_len, obj_height)
                        self.change_state("movedToRecogArea")

                    ## Change grasp if failed, set to std value if success
                    if not succes_grasp:
                        self.grasp_idx += 1
                        self.change_state("graspFailed")
                    elif succes_grasp:
                        self.grasp_idx = 0
                    
                    ##########################################################################
                    ## PERFORMANCE ANALYSIS
                    ##########################################################################

                    print("\nPerformance print, state: ", self.state)
                    print("succes_grasp {}\nsucces_target {}".format(succes_grasp, succes_target))

                    ## TODO: look at target inst below and if performance is saved correctly 
                    data.add_try(target)
                    
                    if succes_grasp:
                        data.add_succes_grasp(target)
                    if succes_target:
                        data.add_succes_target(target)

                    ## remove visualized grasp configuration 
                    if vis:
                        self.remove_drawing(lineIDs)
                        self.remove_drawing(objectTexts)
                        self.remove_drawing(visualTargetBox)

                    env.reset_robot()
                    
                    if succes_target:
                        number_of_failures = 0
                        if vis: self.write_temp_text("succes")
                        
                        if save_name is not None:
                            os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{i}.png')
                        
                    else:
                        number_of_failures += 1                    
                        if vis: self.write_temp_text("failed", [0.5,0,0])

        data.write_json(self.network_model)
        summarize(data.save_dir, runs, self.network_model)


    def piled_target_scenario(self,runs, device, vis, output, debug):
            model, class_names = setup_mrcnn('custom', 'tex/tex100_800st2_endEp30_val0.24/weights.bestVal.hdf5', 0.85)
            objects = YcbObjects('objects/ycb_objects',
                                mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                                mod_stiffness=['Strawberry'])

            
            ## reporting the results at the end of experiments in the results folder
            data = IsolatedObjData(objects.obj_names, runs, 'results')

            ## camera settings: cam_pos, cam_target, near, far, size, fov
            center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
            camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)
            mrcnn_cam = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (448, 448), 40)
            env = Environment(camera, vis=vis, debug=debug, finger_length=0.06)
            
            generator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)

            targettext = ""
            objects.shuffle_objects()
            target_list = objects.obj_names.copy()
            
            ## remove crackerbox since it is ungraspable on its side
            target_list.remove("CrackerBox")

            for targetName in target_list:
                self.state = "idle"

                for i in range(runs):
                    # print("----------- run ", i+1, " -----------")
                    # print ("network model = ", self.network_model)
                    # print("\nTarget object = ", target)
                    if vis: targettext = self.write_perm_text(targettext, "Target: {}".format(targetName))
                    
                    ## Shuffle to-be-spawned-objects and remove target so as not to spawn twice
                    objects.shuffle_objects()
                    other_obj = objects.obj_names.copy()
                    other_obj.remove(targetName)

                    env.reset_robot()          
                    env.remove_all_obj()                        

                    number_of_objects = 5
                    path, mod_orn, mod_stiffness = objects.get_obj_info(targetName)
                    env.load_isolated_obj(path, mod_orn, mod_stiffness)

                    info = objects.get_n_first_obj_info(number_of_objects, targetName)
                    env.create_pile(info)

                    # print("Other objects: {}".format(spawn_obj[1:]))

                    self.dummy_simulation_steps(20)

                    number_of_attempts = self.ATTEMPTS
                    number_of_failures = 0

                    objectTexts = []
                    visualTargetBox = []
                    targetDelivered = False
                    expFailed = False

                    self.grasp_idx = 0 ## select the best grasp configuration
                    failed_to_find_grasp_count = 0
                    min_conf = 0.85

                    while self.is_there_any_object(camera) and number_of_failures < number_of_attempts and targetDelivered != True and expFailed != True:     
                        print("\n--------------------------")
                        rgb, depth, _ = camera.get_cam_img()

                        ##########################################################################
                        ## RECOGNITION
                        ##########################################################################

                        mrcnnRGB, _, _ = mrcnn_cam.get_cam_img()
                        bbox = []
                        mask = []

                        recogObjects, targetIndex, objectTexts = self.run_mrcnn(model, class_names, mrcnnRGB, min_conf, targetName, True)

                        ## Target is found on the table, find best grasp point inside bounding box    
                        if (self.state == "targetFound"):
                            self.change_state("targetGrasp")
                            graspObject = recogObjects[targetIndex]
                            mrcnnBox = graspObject["box"]
                            bbox = (mrcnnBox/(mrcnn_cam.width/camera.width)).astype(int)                ## Resize to 224 for GR ConvNet
                            mask = graspObject["mask"]
                            print('\nTARGET {} FOUND'.format(targetName))                            
                            if vis: 
                                visualTargetBox = self.draw_box(bbox, 224,[0,0.5,0])
                                targettext = self.write_perm_text(targettext, "{} found".format(targetName), [0,0.5,0])

                        ## At least one object found on table, grasp non-target object with highest score
                        elif (self.state == "nonTargetFound"):
                            self.change_state("nonTargetGrasp")
                            graspObject = list(recogObjects.values())[0]      ## use values since there might be a removed crackerbox index
                            mrcnnBox = graspObject["box"]
                            bbox = (mrcnnBox/(mrcnn_cam.width/camera.width)).astype(int)         ## Resize to 224 for GR ConvNet
                            mask = graspObject["mask"]
                            print("\nTarget not found, removing {}".format(graspObject["name"]))
                   
                        ## Object has been moved to recognition area and still not recognized, 
                        ## Lower confidence and restart loop
                        elif (self.state == "movedToRecogArea"):
                            min_conf -= 0.1
                            print("Object has been moved to recog area, and still not recognized.")
                            print("Lowering detection confidence with 0.1 to. ", min_conf)
                            continue
                            
                        ## No object is found on table, freely chosen grasp towards recognition area
                        else:
                            self.change_state("nothingFound")
                            print("\nNo object found on table")
                            if vis: 
                                self.write_temp_text("Object(s) on table, but not recognized", [0.5,0,0])
                                self.write_temp_text("Moving object to recognition area", [0.5,0,0])

                        ##########################################################################
                        ## GRASPING
                        ##########################################################################
                        
                        # if self.state == "targetGrasp" or self.state == "nonTargetGrasp":
                        #     if self.masks_intersect(graspObject,recogObjects):
                        #         print("graspObject mask overlaps with other mask")
                        #         self.isolate_object(graspObject["box"],env)
                        #         if vis: 
                        #             self.remove_drawing(objectTexts)
                        #             targettext = self.write_perm_text(targettext, "Target: {}".format(targetName))
                        #         continue
                                
                        #     else: 
                        #         print("graspObject mask is free")


                            # if self.object_is_isolated(graspObject["box"],graspObject["name"],recogObjects): 
                            #     print("target is isolated")
                            # else: 
                            #     print("target NOT isolated")
                            #     self.isolate_object(graspObject["box"],env)
                            #     if vis: 
                            #         self.remove_drawing(objectTexts)
                            #         targettext = self.write_perm_text(targettext, "Target: {}".format(targetName))
                            #     continue

                        ## Grasp from bounding box, if empty, grasp is freely chosen
                        grasps, save_name = generator.predict_grasp(rgb, depth, bbox, mask, n_grasps=number_of_attempts, show_output=output)

                        ## NO GRASP POINT FOUND
                        if (grasps == []):
                            self.dummy_simulation_steps(50)

                            if self.state == "targetGrasp" or self.state == "nonTargetGrasp":
                                print("No grasp found, increasing size bounding box...")
                                while self.object_is_isolated(mrcnnBox, graspObject["name"], recogObjects) and mrcnnBox.shape < (448,448):
                                    mrcnnBox = self.add_padding_to_box(2,mrcnnBox)
                                    bbox = self.add_padding_to_box(1, bbox)
                                bbox = self.add_padding_to_box(-1, bbox)            ## Remove 1 so that object is still isolated

                            if failed_to_find_grasp_count > 3:
                                print("Failed to find a grasp points > 3 times. Skipping.")
                                if vis:
                                    self.remove_drawing(lineIDs)
                                    self.remove_drawing(objectTexts)
                                    self.remove_drawing(visualTargetBox)
                                break

                            ## Try again to find grasp (do not use mask in this case), if not continue 
                            grasps, save_name = generator.predict_grasp(rgb, depth, bbox, n_grasps=number_of_attempts, show_output=output)
                            if (grasps == []):
                                print("Grasp still empty")
                                self.isolate_object(graspObject["box"],env)

                                if vis: 
                                    self.remove_drawing(objectTexts)
                                    targettext = self.write_perm_text(targettext, "Target: {}".format(targetName))
                                    
                                failed_to_find_grasp_count += 1                 
                                continue

                        elif self.state == "targetGrasp" or self.state == "nonTargetGrasp":
                            ## Check if grasp overlaps with other object
                            self.other_objects_grasped(grasps[self.grasp_idx], graspObject, recogObjects)                     

                        ## idx is iterated after incorrect grasp
                        ## check if this next grasp is possible
                        if (self.grasp_idx > len(grasps)-1):  
                            if len(grasps) > 0 :
                                self.grasp_idx = len(grasps)-1
                            else:
                                number_of_failures += 1
                                continue

                        ## DRAWING GRASP
                        if vis:
                            LID =[]
                            for g in grasps:
                                LID = self.draw_predicted_grasp(g,color=[1,0,1],lineIDs=LID)
                            time.sleep(0.5)
                            self.remove_drawing(LID)
                            self.dummy_simulation_steps(10)    
                        lineIDs = self.draw_predicted_grasp(grasps[self.grasp_idx])


                        x, y, z, yaw, opening_len, obj_height = grasps[self.grasp_idx]
                        # print("Performing final grasp, state: {}".format(self.state))
                        
                        ## Target found, move item to target tray
                        if self.state == "targetGrasp":
                            succes_grasp, succes_target, succes_object = env.targeted_grasp((x, y, z), yaw, opening_len, obj_height, targetName)
                            # print("Succesfully grasped target object == {}".format(succes_object))
                            if succes_target:
                                self.write_temp_text("Target dropped successfully")
                                targetDelivered = True
                            else:
                                targettext = self.write_perm_text(targettext, "Target: {}".format(targetName))

                        ## Non-target object recognized, move item to red tray
                        elif self.state == "nonTargetGrasp":
                            succes_grasp, succes_target, succes_object = env.non_target_grasp((x, y, z), yaw, opening_len, obj_height, targetName)
                            # print("Succesfully grasped nontarget object == {}".format(succes_object))

                        ## No object recognized, move to analysis area
                        elif self.state == "nothingFound":
                            succes_grasp, succes_target = env.move_to_recog_area((x, y, z), yaw, opening_len, obj_height)
                            self.change_state("movedToRecogArea")

                        ## Change grasp if failed, set to std value if success
                        if not succes_grasp:
                            self.grasp_idx += 1
                            self.change_state("graspFailed")
                        elif succes_grasp:
                            self.grasp_idx = 0
                        
                        ##########################################################################
                        ## PERFORMANCE ANALYSIS
                        ##########################################################################

                        # print("\nPerformance print, state: ", self.state)
                        # print("succes_grasp {}\nsucces_target {}".format(succes_grasp, succes_target))

                        ## TODO: look at target inst below and if performance is saved correctly 
                        data.add_try(targetName)
                        
                        if succes_grasp:
                            data.add_succes_grasp(targetName)
                        if succes_target:
                            data.add_succes_target(targetName)

                        ## remove visualized grasp configuration 
                        if vis:
                            self.remove_drawing(lineIDs)
                            self.remove_drawing(objectTexts)
                            self.remove_drawing(visualTargetBox)

                        env.reset_robot()
                        
                        if succes_target:
                            number_of_failures = 0
                            if vis: self.write_temp_text("succes")
                            
                            if save_name is not None:
                                os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{i}.png')
                            
                        else:
                            number_of_failures += 1                    
                            if vis: self.write_temp_text("failed", [0.5,0,0])

            data.write_json(self.network_model)
            summarize(data.save_dir, runs, self.network_model)
    
        
def parse_args():
    parser = argparse.ArgumentParser(description='Grasping demo')
    
    ## for adding terminal command like: 'python simulation.py train'
    ## catch with if args.command == 'train' in __main__
    parser.add_argument("command", metavar="<command>", help="'mask' or 'banana'")
    
    parser.add_argument('--scenario', type=str, default='isolated', help='Grasping scenario (isolated/packed/pile)')
    parser.add_argument('--network', type=str, default='GR_ConvNet', help='Network model (GR_ConvNet/...)')

    parser.add_argument('--runs', type=int, default=1, help='Number of runs the scenario is executed')
    parser.add_argument('--attempts', type=int, default=3, help='Number of attempts in case grasping failed')

    parser.add_argument('--save-network-output', dest='output', type=bool, default=False,
                        help='Save network output (True/False)')

    parser.add_argument('--device', type=str, default='cpu', help='device (cpu/gpu)')
    parser.add_argument('--vis', type=bool, default=True, help='vis (True/False)')
    parser.add_argument('--report', type=bool, default=True, help='report (True/False)')
    parser.add_argument('--colab', type=bool, default=True, help='colab (True/False)')
    parser.add_argument('--background', type=str, default='plain', help='background (jitter / texture)')

                        
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    output = args.output
    runs = args.runs
    ATTEMPTS = args.attempts
    device=args.device
    vis=args.vis
    report=args.report
    colab=args.colab
    background = args.background
    
    if args.command == 'banana':
        look_at_banana(vis)
    elif args.command == 'data':
        make_data(colab, background)
    elif args.command == 'obj':
        look_at_object(vis)
    elif args.command == 'grasp':
        grasp = GrasppingScenarios(args.network)

        if args.scenario == 'isolated':
            grasp.isolated_obj_scenario(runs, device, vis, output=output, debug=False)
        elif args.scenario == 'packed':
            grasp.packed_or_pile_scenario(runs, args.scenario, device, vis, output=output, debug=False)
        elif args.scenario == 'pile':
            grasp.packed_or_pile_scenario(runs, args.scenario, device, vis, output=output, debug=False)
        elif args.scenario == 'iso_target':
            grasp.isolated_target_scenario(runs, device, vis, output=output, debug=False)
        elif args.scenario == 'pile_target':
            grasp.piled_target_scenario(runs, device, vis, output, debug=False)
            
    elif args.command == 'pile':
        grasp = GrasppingScenarios(args.network)
        grasp.piled_target_scenario(runs, device, vis, output, debug=False)

