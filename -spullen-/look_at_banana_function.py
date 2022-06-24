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