from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.env import Environment
from utils import YcbObjects, PackTargetData, PileTargetData, IsolatedTargetData
import numpy as np
import pybullet as p
import argparse
import os
import sys
import math
import matplotlib.pyplot as plt
import time
import random

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.coco import coco

# Root directory of the project
ROOT_DIR = os.path.abspath('/home/ivar/Documents/Thesis/clutterbot/')

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
       
    def setup_mrcnn(self, weights, weights_name, conf = 0.9):
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
        # print(f"\nRecog phase - conf threshold: {min_conf} - ", end= " ")
        start = time.time()

        recogObjects = {}
        targetIndex = np.NaN
        objectTexts = []

        results = model.detect([rgb], verbose=0)
        r = results[0]
        box, mask, classIDs, score = r['rois'], r['masks'], r['class_ids'], r['scores']

        # visualize.display_instances(rgb, box, mask, classIDs, class_names, score)
        end = time.time()
        # print('exec time: {:.2f}'.format(end - start))

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
            # print("{} {:.2f} | ".format(class_names[classIDs[j]], score[j]), end=" ")
        # print("")

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
    
    def box_is_correct(self,box):
        y1, x1, y2, x2 = box

        if y1 < 0:
            print("box check fail, y1 < 0")
            return False
        elif x1 < 0:
            print("box check fail, x1 < 0")
            return False
        elif y2 > 447:
            print("box check fail, y2 > 447")
            return False
        elif x2 > 447:
            print("box check fail, x2 > 447")
            return False
        elif y1 > y2:
            print("box check fail, y1 > y2")
            return False
        elif x1 > x2:
            print("box check fail, x1 > x2")
            
            # classes = ['BG', 'Banana', 'ChipsCan', 'CrackerBox', 'FoamBrick', 'GelatinBox', 'Hammer', 
            #     'MasterChefCan', 'MediumClamp', 'MustardBottle', 'Pear', 'PottedMeatCan', 'PowerDrill', 
            #     'Scissors', 'Strawberry', 'TennisBall', 'TomatoSoupCan']
            # for id in self.temp_r["class_ids"]:
            #     print(classes[id] + " ", end = "")
            # print("boxes: ", self.temp_r["rois"])
            # visualize.display_instances(self.temp_image, self.temp_r['rois'], self.temp_r['masks'], self.temp_r['class_ids'], classes, self.temp_r['scores'])
            
            return False
        else:
            return True
        
    def object_is_isolated(self, box, obj_in_box, recogObjects):
        print("Object isolation check")

        y1, x1, y2, x2 = box
        boxMatrix = np.zeros((448,448))
        
        if self.box_is_correct(box):
            boxMatrix[y1:y2,x1:x2] = 1
            for obj in recogObjects:
                intersect = boxMatrix*obj["mask"]
                if intersect.any() and obj["name"] != obj_in_box:
                    return False
            return True
        else:
            print("Box incorrect, returning True")
            return True
    
    def masks_intersect(self, graspObject, recogObjects):
        for obj in recogObjects.values():
            # print("checking intersection for: ", obj["name"])
            if obj["name"] != graspObject["name"]:
                intersect = obj["mask"]*graspObject["mask"]
                if intersect.any():
                    # print("Intersection found between masks")
                    return True
                    # print("Masks are completely free of intersections")
        return False

    def other_objects_grasped(self, grasp, graspObject, recogObjects):
        # print("Grasp collision check")

        x,y,z, yaw, opening_len, obj_height = grasp
        gripper_size = opening_len + 0.02
        x1 = x+gripper_size*math.sin(yaw)
        x2 = x-gripper_size*math.sin(yaw)
        y1 = y+gripper_size*math.cos(yaw)
        y2 = y-gripper_size*math.cos(yaw)

        y1,x1,y2,x2 = self.transform_meters([y1,x1,y2,x2])

        graspMatrix = np.zeros((448,448))
        graspMatrix[y1:y2,x1:x2] = 1

        for obj in recogObjects.values():
            intersect = graspMatrix*obj["mask"]
            if intersect.any() and obj["name"] != graspObject["name"]:
                return True
        # print("Grasp free of other objects")
        return False

    def target_scenario(self,runs, device, vis, output, scenario, debug):
            model, class_names = self.setup_mrcnn('custom', 'tex/tex100_800st2_endEp30_val0.24/weights.bestVal.hdf5', 0.45)
            objects = YcbObjects('objects/ycb_objects',
                                mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                                mod_stiffness=['Strawberry'])

            ## camera settings: cam_pos, cam_target, near, far, size, fov
            center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
            camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)
            mrcnn_cam = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (448, 448), 40)
            env = Environment(camera, vis=vis, debug=debug, finger_length=0.06)
            generator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)
            
            pile = False
            targettext = ""
            objects.shuffle_objects()
            target_list = objects.obj_names.copy()
            
            if scenario == 'isolated':
                data = IsolatedTargetData(4, 'results')

            elif scenario == 'packed':
                number_of_objects = 4
                data = PackTargetData(number_of_objects, 'results')

            elif scenario == 'pile':
                pile = True
                number_of_objects = 5
                data = PileTargetData(number_of_objects, 'results')

                maskMax = { 'Banana': 2248, 'ChipsCan': 5996, 'FoamBrick': 1447, 'GelatinBox': 2071, 'Hammer': 4112, 'MasterChefCan': 2376, 'MediumClamp': 2373,
                            'MustardBottle': 4812, 'Pear': 1575, 'PottedMeatCan': 2720, 'PowerDrill': 6054, 'Scissors': 2592, 'Strawberry': 1343,
                            'TennisBall': 1148, 'TomatoSoupCan': 2328
                        }
                ## remove crackerbox since it is ungraspable on its side
                target_list.remove("CrackerBox")

            print("targetList = ", target_list)

            for targetName in target_list:
                self.state = "idle"
                print("\n--------------------------")
                data.set_target(targetName)
                print("Target ({}/{})".format(target_list.index(targetName)+1,len(target_list)))

                for i in range(runs):
                    data.new_run(i+1)
                    print("\nrun = ", i+1)

                    if vis: targettext = self.write_perm_text(targettext, "Target: {}".format(targetName))
                    
                    ## Shuffle to-be-spawned-objects and remove target so as not to spawn twice
                    objects.shuffle_objects()
                    other_obj = objects.obj_names.copy()
                    other_obj.remove(targetName)

                    env.reset_robot()          
                    env.remove_all_obj()                        
                    
                    if scenario == 'isolated':
                        spawn_obj = [targetName] + other_obj[0:3]
                        self.spawn_four_objects(objects, spawn_obj, env)
                        data.set_other_obj(spawn_obj[1:])

                    if scenario == 'packed':
                        targetInfo = objects.get_obj_info(targetName)
                        info = objects.get_n_first_obj_info(number_of_objects, targetName)
                        env.create_packed(info, targetInfo)
                        if 'CrackerBox' in env.obj_names: data.set_crackerbox_true()

                    elif scenario == 'pile':
                        path, mod_orn, mod_stiffness = objects.get_obj_info(targetName)
                        env.load_isolated_obj(path, mod_orn, mod_stiffness)

                        _,_,targetSeg = mrcnn_cam.get_cam_img()
                        unique, _ = np.unique(targetSeg, return_counts=True)
                        maskValue = unique[2]

                        info = objects.get_n_first_obj_info(number_of_objects, targetName)
                        env.create_pile(info)
                        if 'CrackerBox' in env.obj_names: data.set_crackerbox_true()

                        _,_,totalSeg = mrcnn_cam.get_cam_img()
                        newMaskCount = np.count_nonzero(totalSeg == maskValue)
                        visibility = round((newMaskCount / maskMax[targetName]), 2)
                        data.set_visibility(visibility)
                        print("Target visibility: ", visibility)

                    
                    # print("Other objects: {}".format(spawn_obj[1:]))

                    self.dummy_simulation_steps(20)

                    number_of_attempts = self.ATTEMPTS
                    number_of_failures = 0

                    objectTexts = []
                    visualTargetBox = []
                    targetDelivered = False
                    expFailed = False
                    expSuccess = False
                    isolation_steps = 0

                    self.grasp_idx = 0 ## select the best grasp configuration
                    failed_to_find_grasp_count = 0
                    min_conf = 0.85

                    while expSuccess != True and expFailed != True:
                        try:     
                            # print("\n--------------------------")
                            rgb, depth, _ = camera.get_cam_img()

                            ## check if target fell off the table, if so stop experiment
                            # target_pos = env.obj_positions[0]
                            # if target_pos[0] < -0.35 or target_pos[0] > 0.45 or target_pos[0][1] > -0.12 or target_pos[0][1] < -0.92:
                            #     print("Target fell on floor, FAIL")
                            #     data.target_fell_off_table()
                            #     break

                            ##########################################################################
                            ## RECOGNITION
                            ##########################################################################

                            mrcnnRGB, _, _ = mrcnn_cam.get_cam_img()
                            bbox = []
                            mask = []

                            recogObjects, targetIndex, objectTexts = self.run_mrcnn(model, class_names, mrcnnRGB, min_conf, targetName, pile)

                            ## Target is found on the table, find best grasp point inside bounding box    
                            if (self.state == "targetFound"):
                                self.change_state("targetGrasp")
                                graspObject = recogObjects[targetIndex]
                                mrcnnBox = graspObject["box"]
                                bbox = (mrcnnBox/(mrcnn_cam.width/camera.width)).astype(int)                ## Resize to 224 for GR ConvNet
                                mask = graspObject["mask"]
                                # print('\nTARGET {} FOUND'.format(targetName))                            
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
                                # print("\nTarget not found, removing {}".format(graspObject["name"]))
                    
                            ## Object has been moved to recognition area and still not recognized, 
                            ## Lower confidence and restart loop
                            elif (self.state == "movedToRecogArea"):
                                min_conf -= 0.1
                                data.lower_conf()
                                # print("Object has been moved to recog area, and still not recognized.")
                                # print("Lowering detection confidence with 0.1 to. ", min_conf)
                                if min_conf < 0.45:
                                    print("confidence lower than detection MRCNN, FAIL")
                                    data.confidence_too_low()
                                    expFailed = True
                                else:
                                    continue
                                
                            ## No object is found on table, freely chosen grasp towards recognition area
                            else:
                                self.change_state("nothingFound")
                                print("No object found on table")
                                if vis: 
                                    self.write_temp_text("Object(s) on table, but not recognized", [0.5,0,0])
                                    self.write_temp_text("Moving object to recognition area", [0.5,0,0])

                            ##########################################################################
                            ## GRASP ANALYSIS
                            ##########################################################################

                            ## Grasp from bounding box, if empty, grasp is freely chosen
                            grasps, save_name = generator.predict_grasp(rgb, depth, bbox, mask, n_grasps=number_of_attempts, show_output=output)

                            ## NO GRASP POINT FOUND
                            if (grasps == []):
                                print("grasps == []")
                                self.dummy_simulation_steps(50)

                                if failed_to_find_grasp_count > 3:
                                    if vis:
                                        self.remove_drawing(lineIDs)
                                        self.remove_drawing(objectTexts)
                                        self.remove_drawing(visualTargetBox)

                                    print("Failed to find grasp > 3 times, FAIL")
                                    data.failed_to_find_grasps()
                                    break

                                if self.state == "targetGrasp" or self.state == "nonTargetGrasp":
                                    self.isolate_object(graspObject["box"],env)
                                    isolation_steps += 1
                                    # self.grasp_idx = 0
                                    data.isolate()

                                if vis: 
                                    self.remove_drawing(objectTexts)
                                    targettext = self.write_perm_text(targettext, "Target: {}".format(targetName))
                                
                                data.empty_grasps()
                                failed_to_find_grasp_count += 1                 
                                continue

                            else:
                                failed_to_find_grasp_count = 0                    

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

                            ##########################################################################
                            ## GRASPING STAGE
                            ##########################################################################

                            x, y, z, yaw, opening_len, obj_height = grasps[self.grasp_idx]
                            # print("Performing final grasp, state: {}".format(self.state))
                            
                            ## Target found, move item to target tray
                            if self.state == "targetGrasp":
                                succes_grasp, succes_tray, succes_object = env.targeted_grasp((x, y, z), yaw, opening_len, obj_height, targetName)
                                # print("Succesfully grasped target object == {}".format(succes_object))
                                if succes_tray:
                                    if succes_object:
                                        self.write_temp_text("Target dropped successfully")
                                        targetDelivered = True
                                    else:
                                        print("nonTarget in target tray, FAIL")
                                        data.wrong_object_in_targetTray(graspObject["name"])
                                        expFailed = True
                                else:
                                    targettext = self.write_perm_text(targettext, "Target: {}".format(targetName))

                            ## Non-target object recognized, move item to red tray
                            elif self.state == "nonTargetGrasp":
                                succes_grasp, succes_tray, succes_object = env.non_target_grasp((x, y, z), yaw, opening_len, obj_height, targetName)
                                if succes_tray:
                                    if succes_object:
                                        data.nonTarget_in_tray()
                                    else:
                                        print("target in wrong tray, FAIL")
                                        data.target_in_nonTarget_tray(graspObject["name"])
                                        expFailed = True

                                # print("Succesfully grasped nontarget object == {}".format(succes_object))

                            ## No object recognized, move to analysis area
                            elif self.state == "nothingFound":
                                succes_grasp, succes_tray = env.move_to_recog_area((x, y, z), yaw, opening_len, obj_height)
                                self.change_state("movedToRecogArea")
                                data.recog_area_move()

                            ## Change grasp if failed, set to std value if success
                            if not succes_grasp:
                                data.grasp(False)
                                if self.state == "targetGrasp" or self.state == "nonTargetGrasp":
                                    if self.grasp_idx + 1 < len(grasps):
                                        ## if next grasp hits other object's mask, isolate object
                                        if self.other_objects_grasped(grasps[self.grasp_idx+1], graspObject, recogObjects):
                                        # if self.masks_intersect(graspObject,recogObjects):
                                            # print("Failed grasp, next grasp hits mask... Isolating object")
                                            self.isolate_object(graspObject["box"],env)
                                            isolation_steps += 1
                                            self.grasp_idx = 0
                                            data.isolate()
                                    elif self.grasp_idx == len(grasps)-1:
                                        ## TODO: make isolation move here??
                                        print("Grasp idx hits end length")
                                        self.isolate_object(graspObject["box"],env)
                                        isolation_steps += 1
                                        self.grasp_idx = 0
                                        data.isolate()
                                        pass
                                    
                                self.grasp_idx += 1
                                # self.change_state("graspFailed")
                            elif succes_grasp:
                                data.grasp(True)
                                self.grasp_idx = 0
                            
                            ##########################################################################
                            ## PERFORMANCE ANALYSIS
                            ##########################################################################

                            # print("\nPerformance print, state: ", self.state)
                            # print("succes_grasp {}\nsucces_target {}".format(succes_grasp, succes_target))
                            if expFailed == False:
                                if targetDelivered:
                                    data.success()
                                    expSuccess = True
                                elif not self.is_there_any_object(camera):
                                    ## if no object is on the table and target is not dropped in tray: exp failed
                                    print("table cleared target not in tray, FAIL")
                                    data.table_clear()
                                    expFailed = True
                                elif number_of_failures >= number_of_attempts:
                                    print("Too many failures, FAIL")
                                    data.fail_count_reached()
                                    expFailed = True
                                elif env.obj_positions[0][2] < 0.2:
                                    print("Target fell on floor, FAIL")
                                    data.target_fell_off_table()
                                    expFailed = True
                                elif isolation_steps > 10:
                                    print("Too many isolation steps, FAIL")
                                    data.isolation_step_limit()
                                    expFailed = True

                            ## remove visualized grasp configuration 
                            if vis:
                                self.remove_drawing(lineIDs)
                                self.remove_drawing(objectTexts)
                                self.remove_drawing(visualTargetBox)

                            env.reset_robot()
                            
                            if succes_tray:
                                data.tray_reached(True)
                                number_of_failures = 0
                                if vis: self.write_temp_text("succes")
                                
                                if save_name is not None:
                                    os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{i}.png')
                                
                            else:
                                data.tray_reached(False)
                                # number_of_failures += 1                    
                                if vis: self.write_temp_text("failed", [0.5,0,0])


                        ##########################################################################

                        except Exception as e:
                                    print("An exception occurred during the experiment!!!")
                                    print(e)

                                    # extensive error reporting (beetje mee oppassen om sys.exc_info() dingen)
                                    # exc_type, exc_obj, exc_tb = sys.exc_info()
                                    # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                    # print(exc_type, fname, exc_tb.tb_lineno)

                                    env.reset_robot()
                                    
            data.save()
            data.print()    
        
def parse_args():
    parser = argparse.ArgumentParser(description='Grasping demo')
    
    ## for adding terminal command like: 'python simulation.py train'
    ## catch with if args.command == 'train' in __main__
    parser.add_argument("command", metavar="<command>", help="'mask' or 'banana'")
    
    parser.add_argument('--scenario', type=str, default='isolated', help='Grasping scenario (isolated/packed/pile)')
    parser.add_argument('--network', type=str, default='GR_ConvNet', help='Network model (GR_ConvNet/...)')

    parser.add_argument('--runs', type=int, default=1, help='Number of runs the scenario is executed')
    parser.add_argument('--attempts', type=int, default=10, help='Number of attempts in case grasping failed')

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
    
    if args.command == 'iso':
        grasp = GrasppingScenarios(args.network)
        grasp.target_scenario(runs, device, vis, output, scenario='isolated', debug=False)

    elif args.command == 'pack':
        grasp = GrasppingScenarios(args.network)
        grasp.target_scenario(runs, device, vis, output, scenario='packed', debug=False)
    
    elif args.command == 'pile':
        grasp = GrasppingScenarios(args.network)
        grasp.target_scenario(runs, device, vis, output, scenario='pile', debug=False)