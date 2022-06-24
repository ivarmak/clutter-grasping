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