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