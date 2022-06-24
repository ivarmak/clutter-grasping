def piled_target_scenario(self,runs, device, vis, output, debug):
        model, class_names = setup_mrcnn('custom', 'tex/tex100_800st2_endEp30_val0.24/weights.bestVal.hdf5', 0.45)
        objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'])

        number_of_objects = 5
        ## reporting the results at the end of experiments in the results folder
        data = PileTargetData(number_of_objects, 'results')

        maskMax = { 'Banana': 2248, 'ChipsCan': 5996, 'FoamBrick': 1447, 'GelatinBox': 2071, 'Hammer': 4112, 'MasterChefCan': 2376, 'MediumClamp': 2373,
                    'MustardBottle': 4812, 'Pear': 1575, 'PottedMeatCan': 2720, 'PowerDrill': 6054, 'Scissors': 2592, 'Strawberry': 1343,
                    'TennisBall': 1148, 'TomatoSoupCan': 2328
                }

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
                
                path, mod_orn, mod_stiffness = objects.get_obj_info(targetName)
                env.load_isolated_obj(path, mod_orn, mod_stiffness)

                _,_,targetSeg = mrcnn_cam.get_cam_img()
                unique, _ = np.unique(targetSeg, return_counts=True)
                maskValue = unique[2]

                info = objects.get_n_first_obj_info(number_of_objects, targetName)
                env.create_pile(info)

                _,_,totalSeg = mrcnn_cam.get_cam_img()
                newMaskCount = np.count_nonzero(totalSeg == maskValue)
                visibility = round((newMaskCount / maskMax[targetName]), 2)
                data.set_visibility(visibility)
                print("Target visibility: ", visibility)

                if 'CrackerBox' in env.obj_names: data.set_crackerbox_true()
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

                        recogObjects, targetIndex, objectTexts = self.run_mrcnn(model, class_names, mrcnnRGB, min_conf, targetName, True)

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
                            print("grasps == []")
                            self.dummy_simulation_steps(50)

                            # if self.state == "targetGrasp" or self.state == "nonTargetGrasp":
                            #     print("No grasp found, increasing size bounding box...")
                            #     while self.object_is_isolated(mrcnnBox, graspObject["name"], recogObjects) and mrcnnBox.shape < (448,448):
                            #         mrcnnBox = self.add_padding_to_box(2,mrcnnBox)
                            #         bbox = self.add_padding_to_box(1, bbox)
                            #     bbox = self.add_padding_to_box(-1, bbox)            ## Remove 1 so that object is still isolated

                            if failed_to_find_grasp_count > 3:
                                if vis:
                                    self.remove_drawing(lineIDs)
                                    self.remove_drawing(objectTexts)
                                    self.remove_drawing(visualTargetBox)

                                print("Failed to find grasp > 3 times, FAIL")
                                data.failed_to_find_grasps()
                                break

                            ## Try again to find grasp (do not use mask in this case), if not continue 
                            # grasps, save_name = generator.predict_grasp(rgb, depth, bbox, n_grasps=number_of_attempts, show_output=output)
                            # if (grasps == []):
                            # print("Grasp still empty")
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
                        
                        # elif self.state == "targetGrasp" or self.state == "nonTargetGrasp":
                        #     ## Check if grasp overlaps with other object
                        #     self.other_objects_grasped(grasps[self.grasp_idx], graspObject, recogObjects) 

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