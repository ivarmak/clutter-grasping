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
                        # print("\nNo object found on table")
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