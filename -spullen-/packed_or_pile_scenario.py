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