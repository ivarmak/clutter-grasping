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
    print("Recognition phase...")
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

    model, class_names = setup_mrcnn('custom', weights5)

    objects = YcbObjects('objects/ycb_objects',
                        mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                        mod_stiffness=['Strawberry'])

    names = ['BG', 'Banana', 'ChipsCan', 'CrackerBox', 'FoamBrick', 'GelatinBox', 'Hammer', 
                'MasterChefCan', 'MediumClamp', 'MustardBottle', 'Pear', 'PottedMeatCan', 'PowerDrill', 
                'Scissors', 'Strawberry', 'TennisBall', 'TomatoSoupCan']

    obj = 'TomatoSoupCan'

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

    # load turned object
    pitch = bool(random.getrandbits(1))
    roll = bool(random.getrandbits(1))
    env.load_turnable_obj(obj_path, pitch, roll)

    ## load pile of objects
    # number_of_objects = 5
    # objects.shuffle_objects()
    # info = objects.get_n_first_obj_info(number_of_objects)

    # env.create_packed(info)
    # env.create_pile(info)

    rgb, _, seg = camera.get_cam_img()

    box, mask, classID, score = evaluate_mrcnn(model, rgb)
    # print(classID)
    plt.imshow(seg)
    visualize.display_instances(rgb, box, mask, classID, class_names, score)

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
            # you need to add your network here!
            print("The selected network has not been implemented yet!")
            exit() 
        
        
        self.CAM_Z = 1.9
        self.depth_radius = 1
        self.ATTEMPTS = 3
        self.fig = plt.figure(figsize=(10, 10))
       
                
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

    def draw_box(self, box, img_size = 448, color = [0,0,1]):
        if (box == []):
            print("Cannot draw EMPTY BOUNDING BOX\n")
            return box
        
        Z = 0.785 # height of workspace thus z variable line
        lines = []

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
        time.sleep(0.5)
        p.removeUserDebugItem(debugID)

    def dummy_simulation_steps(self,n):
        for _ in range(n):
            p.stepSimulation()

    def write_perm_text(self,previoustext, text, color = [0.5,0,0], loc = [-0.15,-0.52,1.92], textSize=2):
        if previoustext != "": p.removeUserDebugItem(previoustext)
        debugID = p.addUserDebugText(text, loc, color, textSize)
        return debugID

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


    def isolated_target_scenario(self,runs, device, vis, output, debug):
        model, class_names = setup_mrcnn('custom', 'tex/tex100_800st2_endEp30_val0.24/weights.bestVal.hdf5', 0.8)
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

        IDLE, SEARCHING, FOUND, NOTFOUND, GRASPING, DELIVERED = 1,2,3,4,5,6

        for target in target_list:
            state = IDLE

            for i in range(runs):
                print("----------- run ", i+1, " -----------")
                print ("network model = ", self.network_model)
                print("\nTarget object = ", target)
                if vis: targettext = self.write_perm_text(targettext, "Target: {}".format(target))
                
                ## Shuffle to-be-spawned-objects and remove target so as not to spawn twice
                objects.shuffle_objects()
                other_obj = objects.obj_names.copy()
                other_obj.remove(target)

                env.reset_robot()          
                env.remove_all_obj()                        

                spawn_obj = [target] + other_obj[0:3]
                self.spawn_four_objects(objects, spawn_obj, env)
                print("Other objects: ", spawn_obj[1:])

                self.dummy_simulation_steps(20)

                number_of_attempts = self.ATTEMPTS
                number_of_failures = 0
                idx = 0 ## select the best grasp configuration
                failed_grasp_counter = 0

                while self.is_there_any_object(camera) and number_of_failures < number_of_attempts and state != DELIVERED:     
                    
                    rgb, depth, _ = camera.get_cam_img()

                    ##########################################################################
                    ## RECOGNITION
                    ##########################################################################
                    
                    state = SEARCHING

                    mrcnnRGB, _, _ = mrcnn_cam.get_cam_img()   
                    box, mask, classIDs, score = evaluate_mrcnn(model, mrcnnRGB)

                    # visualize.display_instances(mrcnnRGB, box, mask, classIDs, class_names, score)
                    bbox = []
                    targetID = class_names.index(target)
                    targetFound = False

                    

                    recogObjects = {}

                    for j in range(classIDs.size):
                        found_obj = class_names[classIDs[j]]
                        convBox = self.transform_coordinates(box[j])
                        obj = {
                            "id" : classIDs[j],
                            "name": found_obj,
                            "box": box[j],
                            "convBox": convBox,
                            "mask": mask[j],
                            "score": score[j],
                            "text": self.write_perm_text("", class_names[classIDs[j]], [0,0,0.5], [convBox[3],convBox[2],0.85], 1)
                        }
                        recogObjects[j] = obj
                        print("Found: {}, score: {:.2f}".format(found_obj, score[j]))
                        
                    # print(recogObjects)
                    if (targetID in classIDs):
                        print('TARGET {} FOUND'.format(target))
                        targettext = self.write_perm_text(targettext, "{} found".format(target), [0,0.5,0])
                        targetFound = True

                        result = np.where(classIDs == targetID)
                        index = result[0][0]
                        targetBox = box[index]

                        ## Resize to 224 for GR ConvNet
                        bbox = (targetBox/2).astype(int)
                        if vis: visualTargetBox = self.draw_box(bbox, 224,[0,0.5,0])
                    else:
                        if vis: 
                            self.write_temp_text("Not found", [0.5,0,0])
                            self.write_temp_text("Grasping other object", [0.5,0,0])
                        print("Target not found, grasping other object")

                    ##########################################################################
                    ## GRASPING
                    ##########################################################################

                    ## Grasp from bounding box
                    grasps, save_name = generator.predict_grasp(rgb, depth, bbox, n_grasps=number_of_attempts, show_output=output)

                    ## Grasp from table
                    # grasps, save_name = generator.predict_grasp( rgb, depth, n_grasps=number_of_attempts, show_output=output)

                    if (grasps == []):
                        self.dummy_simulation_steps(50)
                        #print ("could not find a grasp point!")
                        if failed_grasp_counter > 3:
                            print("Failed to find a grasp points > 3 times. Skipping.")
                            break
                            
                        failed_grasp_counter += 1                 
                        continue

                    #print ("grasps.length = ", len(grasps))
                    if (idx > len(grasps)-1):  
                        if len(grasps) > 0 :
                            idx = len(grasps)-1
                        else:
                            number_of_failures += 1
                            continue    

                    if vis:
                        LID =[]
                        for g in grasps:
                            LID = self.draw_predicted_grasp(g,color=[1,0,1],lineIDs=LID)
                        time.sleep(0.5)
                        self.remove_drawing(LID)
                        self.dummy_simulation_steps(10)
                        

                    lineIDs = self.draw_predicted_grasp(grasps[idx])

                    x, y, z, yaw, opening_len, obj_height = grasps[idx]

                    if targetFound:
                        ## Move item to target tray
                        succes_grasp, succes_target = env.targeted_grasp((x, y, z), yaw, opening_len, obj_height)
                        if succes_target:
                            self.write_temp_text("Target dropped successfully")
                            state = DELIVERED
                    else:
                        ## Move item to red tray
                        succes_grasp, succes_target = env.grasp((x, y, z), yaw, opening_len, obj_height)

                    ##########################################################################
                    ## PERFORMANCE ANALYSIS
                    ##########################################################################

                    ## TODO: look at target inst below and if performance is saved correctly 
                    data.add_try(target)
                    
                    if succes_grasp:
                        data.add_succes_grasp(target)
                    if succes_target:
                        data.add_succes_target(target)

                    ## remove visualized grasp configuration 
                    if vis:
                        self.remove_drawing(lineIDs)
                        if recogObjects != {}:
                            texts = []
                            for o in recogObjects.values():
                                texts.append(o["text"])
                            self.remove_drawing(texts)
                        if targetFound: 
                            self.remove_drawing(visualTargetBox)

                    env.reset_robot()
                    
                    if succes_target:
                        number_of_failures = 0
                        if vis: self.write_temp_text("succes")
                        
                        if save_name is not None:
                            os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{i}.png')
                        

                    else:
                        number_of_failures += 1
                        idx +=1        
                        #env.reset_robot() 
                        # env.remove_all_obj()                        
                        if vis: self.write_temp_text("failed", [0.5,0,0])

        data.write_json(self.network_model)
        summarize(data.save_dir, runs, self.network_model)


    def isolated_obj_scenario(self,runs, device, vis, output, debug):

        objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'])

        
        ## reporting the results at the end of experiments in the results folder
        data = IsolatedObjData(objects.obj_names, runs, 'results')

        ## camera settings: cam_pos, cam_target, near, far, size, fov
        center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
        camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)
        env = Environment(camera, vis=vis, debug=debug, finger_length=0.06)
        
        generator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)
        
        objects.shuffle_objects()
        for i in range(runs):
            print("----------- run ", i+1, " -----------")
            print ("network model = ", self.network_model)
            print ("size of input image (W, H) = (", self.IMG_SIZE," ," ,self.IMG_SIZE, ")")

            for obj_name in objects.obj_names:
                print(obj_name)

                env.reset_robot()          
                env.remove_all_obj()                        
               
                path, mod_orn, mod_stiffness = objects.get_obj_info(obj_name)
                env.load_isolated_obj(path, mod_orn, mod_stiffness)
                
                self.dummy_simulation_steps(20)

                number_of_attempts = self.ATTEMPTS
                number_of_failures = 0
                idx = 0 ## select the best grasp configuration
                failed_grasp_counter = 0
                while self.is_there_any_object(camera) and number_of_failures < number_of_attempts:     
                    
                    bgr, depth, _ = camera.get_cam_img()
                    ##convert BGR to RGB
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    
                    grasps, save_name = generator.predict_grasp( rgb, depth, n_grasps=number_of_attempts, show_output=output)
                    if (grasps == []):
                        self.dummy_simulation_steps(50)
                        #print ("could not find a grasp point!")
                        if failed_grasp_counter > 3:
                            print("Failed to find a grasp points > 3 times. Skipping.")
                            break
                            
                        failed_grasp_counter += 1                 
                        continue

                    #print ("grasps.length = ", len(grasps))
                    if (idx > len(grasps)-1):  
                        print ("idx = ", idx)
                        if len(grasps) > 0 :
                           idx = len(grasps)-1
                        else:
                           number_of_failures += 1
                           continue    

                    if vis:
                        LID =[]
                        for g in grasps:
                            LID = self.draw_predicted_grasp(g,color=[1,0,1],lineIDs=LID)
                        time.sleep(0.5)
                        self.remove_drawing(LID)
                        self.dummy_simulation_steps(10)
                        

                    lineIDs = self.draw_predicted_grasp(grasps[idx])

                    x, y, z, yaw, opening_len, obj_height = grasps[idx]
                    succes_grasp, succes_target = env.grasp_2((x, y, z), yaw, opening_len, obj_height)

                    data.add_try(obj_name)
                   
                    if succes_grasp:
                        data.add_succes_grasp(obj_name)
                    if succes_target:
                        data.add_succes_target(obj_name)

                    ## remove visualized grasp configuration 
                    if vis:
                        self.remove_drawing(lineIDs)

                    env.reset_robot()
                    
                    if succes_target:
                        number_of_failures = 0
                        if vis:
                            debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0,0.50,0], textSize=2)
                            time.sleep(0.25)
                            p.removeUserDebugItem(debugID)
                        
                        if save_name is not None:
                            os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{i}.png')
                        

                    else:
                        number_of_failures += 1
                        idx +=1        
                        #env.reset_robot() 
                        # env.remove_all_obj()                        
                
                        if vis:
                            debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5,0,0], textSize=2)
                            time.sleep(0.25)
                            p.removeUserDebugItem(debugID)

        data.write_json(self.network_model)
        summarize(data.save_dir, runs, self.network_model)


    def packed_or_pile_scenario(self,runs, scenario, device, vis, output, debug):
        
        ## reporting the results at the end of experiments in the results folder
        number_of_objects = 10
        if scenario=='packed':
            objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'],
                            exclude=['CrackerBox'])
            
            data = PackPileData(number_of_objects, runs, 'results', 'packed')

        elif scenario=='pile':
            objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'],
                            exclude=['CrackerBox'])

            data = PackPileData(number_of_objects, runs, 'results', self.network_model, 'pile')


        center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
        camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)
        env = Environment(camera, vis=vis, debug=debug, finger_length=0.06)
        
        generator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)

        
        for i in range(runs):
            env.remove_all_obj()
            env.reset_robot()              
            print("----------- run ", i+1, " -----------")
            print ("network model = ", self.network_model)
            print ("size of input image (W, H) = (", self.IMG_SIZE," ," ,self.IMG_SIZE, ")")

            
            if vis:
                debugID = p.addUserDebugText(f'Experiment {i+1}', [-0.0, -0.9, 0.8], [0,0,255], textSize=2)
                time.sleep(0.5)
                p.removeUserDebugItem(debugID)

            number_of_failures = 0
            objects.shuffle_objects()

            info = objects.get_n_first_obj_info(number_of_objects)

            if scenario=='packed':
                env.create_packed(info)
            elif scenario=='pile':
                env.create_pile(info)
                
            #self.dummy_simulation_steps(50)

            number_of_failures = 0
            ATTEMPTS = 4
            number_of_attempts = ATTEMPTS
            failed_grasp_counter = 0
            flag_failed_grasp_counter= False

            while self.is_there_any_object(camera) and number_of_failures < number_of_attempts:                
                #env.move_arm_away()
                try: 
                    idx = 0 ## select the best grasp configuration
                    for i in range(number_of_attempts):
                        data.add_try()  
                        rgb, depth, _ = camera.get_cam_img()
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                        
                        grasps, save_name = generator.predict_grasp( rgb, depth, n_grasps=number_of_attempts, show_output=output)
                        if (grasps == []):
                                self.dummy_simulation_steps(30)
                                print ("could not find a grasp point!")    
                                if failed_grasp_counter > 5:
                                    print("Failed to find a grasp points > 5 times. Skipping.")
                                    flag_failed_grasp_counter= True
                                    break

                                failed_grasp_counter += 1 
                                continue
                        
                        if vis:
                            LID =[]
                            for g in grasps:
                                LID = self.draw_predicted_grasp(g,color=[1,0,1],lineIDs=LID)
                            time.sleep(0.5)
                            self.remove_drawing(LID)
                            self.dummy_simulation_steps(10)
                            
                        #print ("grasps.length = ", len(grasps))
                        if (idx > len(grasps)-1):  
                            print ("idx = ", idx)
                            if len(grasps) > 0 :
                                idx = len(grasps)-1
                            else:
                                number_of_failures += 1
                                continue  

                        lineIDs = self.draw_predicted_grasp(grasps[idx])
                        
                        ## perform object grasping and manipulation : 
                        #### succes_grasp means if the grasp was successful, 
                        #### succes_target means if the target object placed in the target basket successfully
                        x, y, z, yaw, opening_len, obj_height = grasps[idx]
                        succes_grasp, succes_target = env.grasp((x, y, z), yaw, opening_len, obj_height)
                        
                        if succes_grasp:
                            data.add_succes_grasp()                     

                        ## remove visualized grasp configuration 
                        if vis:
                            self.remove_drawing(lineIDs)

                        env.reset_robot()
                            
                        if succes_target:
                            data.add_succes_target()
                            number_of_failures = 0

                            if vis:
                                debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0,0.50,0], textSize=2)
                                time.sleep(0.25)
                                p.removeUserDebugItem(debugID)
                                
                            if save_name is not None:
                                    os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{i}.png')
                                
                        else:
                            #env.reset_robot()   
                            number_of_failures += 1
                            idx +=1     
                                                    
                            if vis:
                                debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5,0,0], textSize=2)
                                time.sleep(0.25)
                                p.removeUserDebugItem(debugID)

                            if number_of_failures == number_of_attempts:
                                if vis:
                                    debugID = p.addUserDebugText("breaking point",  [-0.0, -0.9, 0.8], [0.5,0,0], textSize=2)
                                    time.sleep(0.25)
                                    p.removeUserDebugItem(debugID)
                                
                                break

                        #env.reset_all_obj()
        
                except Exception as e:
                    print("An exception occurred during the experiment!!!")
                    print(e)

                    # extensive error reporting (beetje mee oppassen om sys.exc_info() dingen)
                    # exc_type, exc_obj, exc_tb = sys.exc_info()
                    # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    # print(exc_type, fname, exc_tb.tb_lineno)

                    env.reset_robot()
                    #print ("#objects = ", len(env.obj_ids), "#failed = ", number_of_failures , "#attempts =", number_of_attempts)
        
                if flag_failed_grasp_counter:
                    flag_failed_grasp_counter= False
                    break
        data.summarize() 
        
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

