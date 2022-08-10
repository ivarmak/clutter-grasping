def look_at_object(vis):
    CAM_Z = 1.9
    IMG_SIZE = 224
    MRCNN_IMG_SIZE = 448

    weights = 'bestMRCNN_1000st_20ep_augSeg_gt1_val0.19'
    weights2 = 'MRCNN_st300_20ep_augSeq_GT1_val0.18'
    weights3 = 'mask_rcnn_object_0032'
    weights4 = 'rand/rand_4000st/weights.bestVal=0.22.hdf5'
    weights5 = 'tex/tex100_800st2_endEp30_val0.24/weights.bestVal.hdf5'

    # model, class_names = setup_mrcnn('custom', weights5, 0.8)

    objects = YcbObjects('objects/ycb_objects',
                        mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                        mod_stiffness=['Strawberry'])

    names = ['BG', 'Banana', 'ChipsCan', 'CrackerBox', 'FoamBrick', 'GelatinBox', 'Hammer', 
                'MasterChefCan', 'MediumClamp', 'MustardBottle', 'Pear', 'PottedMeatCan', 'PowerDrill', 
                'Scissors', 'Strawberry', 'TennisBall', 'TomatoSoupCan']

    obj = 'MediumClamp'

    obj_path = 'objects/ycb_objects/Ycb' + obj + '/model.urdf'

    ## camera settings: cam_pos, cam_target, near, far, size, fov
    center_x, center_y, center_z = 0.05, -0.52, CAM_Z

    # camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (IMG_SIZE, IMG_SIZE), 40)
    camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (MRCNN_IMG_SIZE, MRCNN_IMG_SIZE), 40)
    env = Environment(camera, vis=True, finger_length=0.06)


    env.reset_robot()          
    env.remove_all_obj()                        
    
    # load object into environment
    # env.load_isolated_obj(obj_path, 0.6)

    #  # # load pile of objects
    # number_of_objects = 5
    # objects.shuffle_objects()
    # info = objects.get_n_first_obj_info(number_of_objects)
    # env.create_pile(info)

    # for _ in range(1000):
    #     p.stepSimulation()

    # print("positions: ", env.obj_positions)
    # print("target position z = ", env.obj_positions[0][2])
    # names = ['Scissors', 'Strawberry', 'TennisBall', 'TomatoSoupCan']
    # for obj in names:
    #     # while inpt != 'n':
    #     print(obj)
    #     highMask = 0
    #     for _ in range(100):
    #         if obj == "BG": continue

    #         ## load turned object
    #         obj_path = 'objects/ycb_objects/Ycb' + obj + '/model.urdf'
    #         pitch = bool(random.getrandbits(1))
    #         roll = bool(random.getrandbits(1))
    #         env.load_turnable_obj(obj_path, pitch, roll)

    #         _,_,targetSeg = camera.get_cam_img()
    #         _, counts = np.unique(targetSeg, return_counts=True)
    #         if len(counts) > 2:
    #             # maskValue = unique[2]
    #             maskCount = counts[2]
    #             # print(obj, maskCount)
    #             if maskCount > highMask:
    #                 highMask = maskCount
    #                 print('new highmask: ', highMask)
    #         # else: print ("tafelstuiter")
    #         # plt.imshow(rgb)
    #         # plt.waitforbuttonpress()

    #         env.remove_all_obj()

    env.load_isolated_obj(obj_path, True)
    target_pos = env.obj_positions[0]
    print(target_pos)
    if target_pos[0] < -0.35 or target_pos[0] > 0.45 or target_pos[0][1] > -0.12 or target_pos[0][1] < -0.92:
        print("Target fell on floor, FAIL")
        # data.target_fell_off_table()
        # break
    # _,_,singleseg = camera.get_cam_img()

    # unique, counts = np.unique(singleseg, return_counts=True)
    # maskvalue = unique[2]
    # maskcount = counts[2]
    # print("\n maskvalue: ", maskvalue)
    # print("maskamount: ", maskcount)

    # # waardes = dict(zip(unique, counts))
    # # print("before dict: ", unique[2], counts[2], "\n")
    # # print("dict = ", waardes)    
    # # nonmask = np.count_nonzero((singleseg == 1) | (singleseg == 6))
    # # print("total values 448x448 = 200704 \n - nonmask values: {} = {}\n".format(nonmask, 200704-nonmask))
    # # maskvalue = list(waardes.keys())[2]
    # # maskamount = waardes[maskvalue]

    

    # # plt.figure()
    # # plt.imshow(singleseg)
    # # plt.waitforbuttonpress()

    # ## load turned object
    # # pitch = bool(random.getrandbits(1))
    # # roll = bool(random.getrandbits(1))
    # # env.load_turnable_obj(obj_path, pitch, roll)

    # # load pile of objects
    # number_of_objects = 5
    # objects.shuffle_objects()
    # info = objects.get_n_first_obj_info(number_of_objects, 'MediumClamp')

    # # env.create_packed(info)
    # env.create_pile(info)

    # rgb, _, seg = camera.get_cam_img()

    # newmaskcount = np.count_nonzero(seg == maskvalue)
    # print("new count: ", newmaskcount)
    # print("fraction = ", newmaskcount/maskcount)

    # # box, mask, classID, score = evaluate_mrcnn(model, rgb)
    # # print(classID)
    # # plt.imshow(seg)

    # # visualize.display_instances(rgb, box, mask, classID, class_names, score)

    # plt.figure()
    # plt.imshow(seg)
    # plt.waitforbuttonpress()
    # # plt.imshow(rgb)
    # # plt.waitforbuttonpress()

    def look_at_object(vis):
        CAM_Z = 1.9
        IMG_SIZE = 224
        MRCNN_IMG_SIZE = 448

        weights = 'bestMRCNN_1000st_20ep_augSeg_gt1_val0.19'
        weights2 = 'MRCNN_st300_20ep_augSeq_GT1_val0.18'
        weights3 = 'mask_rcnn_object_0032'
        weights4 = 'rand/rand_4000st/weights.bestVal=0.22.hdf5'
        weights5 = 'tex/tex100_800st2_endEp30_val0.24/weights.bestVal.hdf5'

        # model, class_names = setup_mrcnn('custom', weights5, 0.8)

        objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'])

        names = ['BG', 'Banana', 'ChipsCan', 'CrackerBox', 'FoamBrick', 'GelatinBox', 'Hammer', 
                    'MasterChefCan', 'MediumClamp', 'MustardBottle', 'Pear', 'PottedMeatCan', 'PowerDrill', 
                    'Scissors', 'Strawberry', 'TennisBall', 'TomatoSoupCan']

        obj = 'FoamBrick'

        obj_path = 'objects/ycb_objects/Ycb' + obj + '/model.urdf'

        ## camera settings: cam_pos, cam_target, near, far, size, fov
        center_x, center_y, center_z = 0.05, -0.52, CAM_Z

        # camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (IMG_SIZE, IMG_SIZE), 40)
        camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (MRCNN_IMG_SIZE, MRCNN_IMG_SIZE), 40)
        env = Environment(camera, vis=True, finger_length=0.06)

        # for _ in range(10):
        #     env.reset_robot()          
        #     env.remove_all_obj()

        #     env.load_obj_same_place(obj_path, -0.2, -0.12)

        #     rgb, _ ,_ = camera.get_cam_img()

        #     ## center point x coordinate is evaluated?

        #     target_pos = env.obj_positions[0]
        #     if target_pos[0] < -0.35 or target_pos[0] > 0.45 or target_pos[1] > -0.12 or target_pos[1] < -0.92:
        #         print("Target fell on floor, FAIL")
        #     else:
        #         print("whithin limits")

        #     plt.imshow(rgb)
        #     plt.waitforbuttonpress()

        while(True):
            pass
