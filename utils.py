import random
from datetime import datetime
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch


class YcbObjects:
    def __init__(self, load_path, mod_orn=None, mod_stiffness=None, exclude=None):
        self.load_path = load_path
        self.mod_orn = mod_orn
        self.mod_stiffness = mod_stiffness
        with open(load_path + '/obj_list.txt') as f:
            lines = f.readlines()
            self.obj_names = [line.rstrip('\n') for line in lines]
        if exclude is not None:
            for obj_name in exclude:
                self.obj_names.remove(obj_name)

    def shuffle_objects(self):
        random.shuffle(self.obj_names)

    def get_obj_path(self, obj_name):
        return f'{self.load_path}/Ycb{obj_name}/model.urdf'

    def check_mod_orn(self, obj_name):
        if self.mod_orn is not None and obj_name in self.mod_orn:
            return True
        return False

    def check_mod_stiffness(self, obj_name):
        if self.mod_stiffness is not None and obj_name in self.mod_stiffness:
            return True
        return False

    def get_obj_info(self, obj_name):
        return self.get_obj_path(obj_name), self.check_mod_orn(obj_name), self.check_mod_stiffness(obj_name)

    def get_n_first_obj_info(self, n, target = ""):
        info = []
        if target != "":
            while(target in self.obj_names[:n]):
                self.shuffle_objects()
        for obj_name in self.obj_names[:n]:
                info.append(self.get_obj_info(obj_name))
        return info

class IsolatedTargetData:

    def __init__(self, num_of_obj, save_path):
        
        self.PRINT = False

        labels = ["Target", "Run", "Success", "FailReason", "NumberOfObjects", "NonTargetsRemoved", "IsolationMoves", 
                    "FailedGrasps", "SuccessGrasps", "FailedTray", "SuccessTray", "NoGraspFound", "ConfScoreEnd",
                     "RecogAreaMoves", "MisclassifiedAs", "NonTargetInTargetTray"]
        self.df = pd.DataFrame(columns= labels)
        

    ## init values
        self.num_of_obj = num_of_obj

    ## values for each run
        self.target = ""
        self.run = 0
        self.target_delivered = False

        self.nonTargets_removed = 0
        self.isolation_moves = 0
        self.failed_grasps = 0
        self.success_grasps = 0
        self.failed_tray = 0
        self.success_tray = 0
        self.no_grasp_found = 0
        self.conf_score_at_end = 0.85
        self.moves_to_recogArea = 0

        ## show stoppers
        self.fail_reason = ""
        self.target_misclassified_as = ""       
        self.nonTarget_in_targetTray = ""       ## string for which object

        self.save_path = save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        now = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        self.save_dir = f'{save_path}/{now}_isolated_target'
        os.mkdir(self.save_dir)

    #########################
    ## Starters
    ######################### 

    def set_target(self, t):
        print("set target, ", t)

        self.target = t

    #########################
    ## Methods
    #########################    

    def new_run(self, r):
        if self.PRINT: print("new run")

        self.run = r
        self.reset_values()
        
    def success(self):
        print("SUCCESS")

        self.target_delivered = True
        self.finish_run()

    def finish_run(self):
        if self.PRINT: print("finish run")

        new_row = self.make_row()
        self.df.loc[len(self.df.index)] = new_row

    def reset_values(self):
        if self.PRINT: print("reset values")

        self.target_delivered = False
        self.nonTargets_removed = 0
        self.isolation_moves = 0
        self.failed_grasps = 0
        self.success_grasps = 0
        self.failed_tray = 0
        self.success_tray = 0
        self.no_grasp_found = 0
        self.conf_score_at_end = 0.85
        self.moves_to_recogArea = 0

        self.fail_reason = ""
        self.target_misclassified_as = ""
        self.nonTarget_in_targetTray = ""
    
    #########################
    ## Numericals
    #########################  

    def empty_grasps(self):
        if self.PRINT: print("empty grasp")

        self.no_grasp_found += 1

    def grasp(self, success):
        if self.PRINT: print("graspsuccess: ", success)
        
        if success: self.success_grasps += 1
        else: self.failed_grasps += 1

    def isolate(self):
        if self.PRINT: print("isolate")

        self.isolation_moves += 1

    def recog_area_move(self):
        if self.PRINT: print("recog area move")

        self.moves_to_recogArea += 1

    def lower_conf(self):
        if self.PRINT: print("lower conf")

        self.conf_score_at_end -= 0.1

    def tray_reached(self, success):
        if self.PRINT: print("traysuccess: ", success)
        
        if success: self.success_tray += 1
        else: self.failed_tray += 1
    
    def nonTarget_in_tray(self):
        if self.PRINT: print("nonTarget in correct tray")

        self.nonTargets_removed += 1

    #########################
    ## Failures
    #########################  

    def table_clear(self):
        if self.PRINT: print("table clear")

        self.fail_reason = "tableCleared"
        self.finish_run()

    def isolation_step_limit(self):
        if self.PRINT: print("isolation step limit")

        self.fail_reason = "tooManyIsolationSteps"
        self.finish_run()

    def confidence_too_low(self):
        if self.PRINT: print("confidence too low")

        self.fail_reason = "confTooLow"
        self.finish_run()

    def wrong_object_in_targetTray(self, obj):
        if self.PRINT: print("wrong object in targetTray: ", obj)
        
        self.fail_reason = "nonTargetinTargetTray"
        self.nonTarget_in_targetTray = obj
        self.finish_run()

    def target_in_nonTarget_tray(self, obj):
        if self.PRINT: print("target in wrong tray")

        self.fail_reason = "targetInNonTargetTray"
        self.target_misclassified_as = obj
        self.finish_run()
    
    def failed_to_find_grasps(self):
        if self.PRINT: print("grasp finding fail")

        self.fail_reason = "graspFindFail"
        self.finish_run()

    def fail_count_reached(self):
        if self.PRINT: print("fail count reached")

        self.fail_reason = "tooManyFailures"
        self.finish_run()

    def target_fell_off_table(self):
        if self.PRINT: print("target on floor")

        self.fail_reason = "targetOnFloor"
        self.finish_run()

    #########################
    ## Saving
    #########################  

    def make_row(self):
        r = [
            self.target,
            self.run,
            self.target_delivered,
            self.fail_reason,
            self.num_of_obj,           
            self.nonTargets_removed,
            self.isolation_moves,
            self.failed_grasps,
            self.success_grasps,
            self.failed_tray,
            self.success_tray,
            self.no_grasp_found,
            self.conf_score_at_end,
            self.moves_to_recogArea,    
            self.target_misclassified_as,     
            self.nonTarget_in_targetTray
        ]
        return r

    def save(self):
        self.df.to_pickle(os.path.join(self.save_dir,'results'))

class PackPileTargetData:

    def __init__(self, num_of_obj, save_path):
        
        self.PRINT = False

        labels = ["Target", "Run", "TargetVisibility", "Success", "FailReason", "CrackerBox", "NumberOfObjects", "NonTargetsRemoved", "IsolationMoves", 
                    "FailedGrasps", "SuccessGrasps", "FailedTray", "SuccessTray", "NoGraspFound", "ConfScoreEnd",
                     "RecogAreaMoves", "MisclassifiedAs", "NonTargetInTargetTray"]
        self.df = pd.DataFrame(columns= labels)
        

    ## init values
        self.num_of_obj = num_of_obj

    ## values for each run
        self.target = ""
        self.run = 0
        self.target_visibility = 0
        self.target_delivered = False

        self.nonTargets_removed = 0
        self.isolation_moves = 0
        self.failed_grasps = 0
        self.success_grasps = 0
        self.failed_tray = 0
        self.success_tray = 0
        self.no_grasp_found = 0
        self.conf_score_at_end = 0.85
        self.moves_to_recogArea = 0
        self.crackerbox_apparent = False

        ## show stoppers
        self.fail_reason = ""
        self.target_misclassified_as = ""       
        self.nonTarget_in_targetTray = ""       ## string for which object

        self.save_path = save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        now = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        self.save_dir = f'{save_path}/{now}_targeted_pile'
        os.mkdir(self.save_dir)

    #########################
    ## Starters
    ######################### 

    def set_target(self, t):
        print("set target, ", t)

        self.target = t
    
    def set_visibility(self, vis):
        self.target_visibility = vis

    def set_crackerbox_true(self):
        self.crackerbox_apparent = True

    #########################
    ## Methods
    #########################    

    def new_run(self, r):
        if self.PRINT: print("new run")

        self.run = r
        self.reset_values()
        
    def success(self):
        print("SUCCESS")

        self.target_delivered = True
        self.finish_run()

    def finish_run(self):
        if self.PRINT: print("finish run")

        new_row = self.make_row()
        self.df.loc[len(self.df.index)] = new_row

    def reset_values(self):
        if self.PRINT: print("reset values")

        self.target_visibility = 0
        self.target_delivered = False
        self.nonTargets_removed = 0
        self.isolation_moves = 0
        self.failed_grasps = 0
        self.success_grasps = 0
        self.failed_tray = 0
        self.success_tray = 0
        self.no_grasp_found = 0
        self.conf_score_at_end = 0.85
        self.moves_to_recogArea = 0

        self.fail_reason = ""
        self.crackerbox_apparent = False
        self.target_misclassified_as = ""
        self.nonTarget_in_targetTray = ""
    
    #########################
    ## Numericals
    #########################  

    def empty_grasps(self):
        if self.PRINT: print("empty grasp")

        self.no_grasp_found += 1

    def grasp(self, success):
        if self.PRINT: print("graspsuccess: ", success)
        
        if success: self.success_grasps += 1
        else: self.failed_grasps += 1

    def isolate(self):
        if self.PRINT: print("isolate")

        self.isolation_moves += 1

    def recog_area_move(self):
        if self.PRINT: print("recog area move")

        self.moves_to_recogArea += 1

    def lower_conf(self):
        if self.PRINT: print("lower conf")

        self.conf_score_at_end -= 0.1

    def tray_reached(self, success):
        if self.PRINT: print("traysuccess: ", success)
        
        if success: self.success_tray += 1
        else: self.failed_tray += 1
    
    def nonTarget_in_tray(self):
        if self.PRINT: print("nonTarget in correct tray")

        self.nonTargets_removed += 1

    #########################
    ## Failures
    #########################  

    def table_clear(self):
        if self.PRINT: print("table clear")

        self.fail_reason = "tableCleared"
        self.finish_run()

    def isolation_step_limit(self):
        if self.PRINT: print("isolation step limit")

        self.fail_reason = "tooManyIsolationSteps"
        self.finish_run()

    def wrong_object_in_targetTray(self, obj):
        if self.PRINT: print("wrong object in targetTray: ", obj)
        
        self.fail_reason = "nonTargetinTargetTray"
        self.nonTarget_in_targetTray = obj
        self.finish_run()

    def target_in_nonTarget_tray(self, obj):
        if self.PRINT: print("target in wrong tray")

        self.fail_reason = "targetInNonTargetTray"
        self.target_misclassified_as = obj
        self.finish_run()
    
    def failed_to_find_grasps(self):
        if self.PRINT: print("grasp finding fail")

        self.fail_reason = "graspFindFail"
        self.finish_run()

    def fail_count_reached(self):
        if self.PRINT: print("fail count reached")

        self.fail_reason = "tooManyFailures"
        self.finish_run()

    def target_fell_off_table(self):
        if self.PRINT: print("target on floor")

        self.fail_reason = "targetOnFloor"
        self.finish_run()

    #########################
    ## Saving
    #########################  

    def make_row(self):
        r = [
            self.target,
            self.run,
            self.target_visibility,
            self.target_delivered,
            self.fail_reason,
            self.crackerbox_apparent,
            self.num_of_obj,           
            self.nonTargets_removed,
            self.isolation_moves,
            self.failed_grasps,
            self.success_grasps,
            self.failed_tray,
            self.success_tray,
            self.no_grasp_found,
            self.conf_score_at_end,
            self.moves_to_recogArea,    
            self.target_misclassified_as,     
            self.nonTarget_in_targetTray
        ]
        return r

    def save(self):
        self.df.to_pickle(os.path.join(self.save_dir,'results'))


class PackPileData:

    def __init__(self, num_obj, trials, save_path, network, scenario):
        self.num_obj = num_obj
        self.trials = trials
        self.save_path = save_path

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        now = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        self.save_dir = f'{save_path}/{network}_{now}_{scenario}'
        os.mkdir(self.save_dir)

        self.tries = 0
        self.succes_grasp = 0
        self.succes_target = 0

    def add_try(self):
        self.tries += 1

    def add_succes_target(self):
        self.succes_target += 1

    def add_succes_grasp(self):
        self.succes_grasp += 1

    def summarize(self):
        grasp_acc = self.succes_grasp / self.tries
        target_acc = self.succes_target / self.tries
        perc_obj_cleared = self.succes_target / (self.trials * self.num_obj)

        with open(f'{self.save_dir}/summary.txt', 'w') as f:
            f.write(
                f'Stats for {self.num_obj} objects out of {self.trials} trials\n')
            f.write(
                f'Manipulation success rate = {target_acc:.3f} ({self.succes_target}/{self.tries})\n')
            f.write(
                f'Grasp success rate = {grasp_acc:.3f} ({self.succes_grasp}/{self.tries})\n')
            f.write(
                f'Percentage of objects removed from the workspace = {perc_obj_cleared} ({self.succes_target}/{(self.trials * self.num_obj)})\n')

       

        # plot the obtained results
        import numpy as np
        results = [grasp_acc, target_acc, perc_obj_cleared]
        metrics = ['grasp', 'manipulation', '% removed from WS']

        x_pos = np.arange(len(metrics))
        
        # Create bars and choose color
        fig,ax = plt.subplots()
        ax.set_axisbelow(True)
        plt.grid(color='#95a5a6', linestyle='-', linewidth=1, alpha=0.25)
        plt.ylim(0, 1)
        # Add title and axis names        
        plt.rc('axes', titlesize=16)     # fontsize of the axes title
        plt.title(f'Summary of performance for {self.trials} runs')
        plt.xlabel('')
        plt.ylabel('Succes rate (%)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)
        plt.xticks(x_pos, metrics)
        plt.bar(x_pos, results)
        # save graph
        plt.savefig(self.save_dir+'/plot.png')

    def summarize_multi_data(self, **kwargs):

        plot_type = kwargs.get('plot_type','single')
        model_name = kwargs.get('m_name')
        
        if plot_type == 'single':
            
            grasp_acc = self.succes_grasp / self.tries
            target_acc = self.succes_target / self.tries
            perc_obj_cleared = self.succes_target / (self.trials * self.num_obj)
            self.data.append([grasp_acc, target_acc, perc_obj_cleared])
            self.models.append(model_name)
            
            with open(f'{self.save_dir}/summary_'+model_name+'.txt', 'w') as f:
                f.write(
                    f'Stats for {self.num_obj} objects out of {self.trials} trials\n')
                f.write(
                    f'Manipulation success rate = {target_acc:.3f} ({self.succes_target}/{self.tries})\n')
                f.write(
                    f'Grasp success rate = {grasp_acc:.3f} ({self.succes_grasp}/{self.tries})\n')
                f.write(
                    f'Percentage of objects removed from the workspace = {perc_obj_cleared} ({self.succes_target}/{(self.trials * self.num_obj)})\n')
            
            results = [grasp_acc, target_acc, perc_obj_cleared]
       

        # plot the obtained results
        import numpy as np
        
        metrics = ['grasp', 'manipulation', 'removed from WS']

        x_pos = np.arange(len(metrics))
        
        # Create bars and choose color
        fig,ax = plt.subplots()
        ax.set_axisbelow(True)
        plt.grid(color='#95a5a6', linestyle='-', linewidth=1, alpha=0.25)
        plt.ylim(0, 1)
        # Add title and axis names        
        plt.rc('axes', titlesize=16)     # fontsize of the axes title
        plt.title(f'Summary of performance for {self.trials} runs')
        plt.xlabel('')
        plt.ylabel('Succes rate (%)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)
        plt.xticks(x_pos, metrics)
        
        if plot_type == 'single':
            plt.bar(x_pos, results)
            # save graph
            plt.savefig(self.save_dir+'/'+model_name+'_plot.png')
        
        if plot_type == 'multiple':
            offset = 0
            print(self.data)
            print(len(self.data))
            for data in range(len(self.data)):
                plt.bar(x_pos+offset, self.data[data], 0.2)
                offset+=0.2
            
            plt.legend(self.models)    
            # save graph
            plt.savefig(self.save_dir+'/m_plot.png')

class IsolatedObjData:

    def __init__(self, obj_names, trials, save_path):
        self.obj_names = obj_names
        self.trials = trials
        self.succes_target = dict.fromkeys(obj_names, 0)
        self.succes_grasp = dict.fromkeys(obj_names, 0)
        self.tries = dict.fromkeys(obj_names, 0)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        now = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        self.save_dir = f'{save_path}/{now}_iso_obj'
        os.mkdir(self.save_dir)

    def add_succes_target(self, obj_name):
        self.succes_target[obj_name] += 1

    def add_succes_grasp(self, obj_name):
        self.succes_grasp[obj_name] += 1

    def add_try(self, obj_name):
        self.tries[obj_name] += 1

    # def write_json(self):
    #     data_tries = json.dumps(self.tries)
    #     data_target = json.dumps(self.succes_target)
    #     data_grasp = json.dumps(self.succes_grasp)
    #     f = open(self.save_dir+'/data_tries.json', 'w')
    #     f.write(data_tries)
    #     f.close()
    #     f = open(self.save_dir+'/data_target.json', 'w')
    #     f.write(data_target)
    #     f.close()
    #     f = open(self.save_dir+'/data_grasp.json', 'w')
    #     f.write(data_grasp)
    #     f.close()

    def write_json(self, modelname):
        data_tries = json.dumps(self.tries)
        data_target = json.dumps(self.succes_target)
        data_grasp = json.dumps(self.succes_grasp)
        f = open(self.save_dir+'/'+modelname+'_data_tries.json', 'w')
        f.write(data_tries)
        f.close()
        f = open(self.save_dir+'/'+modelname+'_data_target.json', 'w')
        f.write(data_target)
        f.close()
        f = open(self.save_dir+'/'+modelname+'_data_grasp.json', 'w')
        f.write(data_grasp)
        f.close()

def plot(path, tries, target, grasp, trials):
    succes_rate = dict.fromkeys(tries.keys())
    for obj in succes_rate.keys():
        t = tries[obj]
        if t == 0:
            t = 1
        acc_target = target[obj] / t
        acc_grasp = grasp[obj] / t
        succes_rate[obj] = (acc_target, acc_grasp)

    plt.rc('axes', titlesize=13)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels

    df = pd.DataFrame(succes_rate).T
    df.columns = ['Manipulation', 'Grasp']
    df = df.sort_values(by='Manipulation', ascending=True)
    ax = df.plot(kind='bar', color=['#88CCEE', '#CC6677'])
    # ax = df.plot(kind='bar', color=['b', 'r'])

    plt.xlabel('objects')
    plt.ylabel('succes rate (%)')
    plt.title(
        f'Succes rate for object grasping and manipulation | {trials} runs')
    plt.grid(color='#95a5a6', linestyle='-', linewidth=1, axis='y', alpha=0.5)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")
    plt.locator_params(axis="y", nbins=11)

    plt.legend(loc='lower right')
    plt.subplots_adjust(bottom=0.28)

    plt.savefig(path+'/plot.png')


def write_summary(path, tries, target, grasp):
    with open(path+'/summary.txt', 'w') as f:
        total_tries = sum(tries.values())
        total_target = sum(target.values())
        total_grasp = sum(grasp.values())
        f.write('Total:\n')
        f.write(
            f'Grasp acc={total_grasp/total_tries:.3f} ({total_grasp}/{total_tries}) --- Manipulation acc={total_target/total_tries:.3f} ({total_target}/{total_tries}) \n')
        f.write('\n')
        f.write("Accuracy per object:\n")
        for obj in tries.keys():
            n_tries = tries[obj]
            n_t = target[obj]
            n_g = grasp[obj]
            f.write(
                f'{obj}: Grasp acc={n_g/n_tries:.3f} ({n_g}/{n_tries}) --- Manipulation acc={n_t/n_tries:.3f} ({n_t}/{n_tries}) \n')


def summarize(path, trials,modelname):
    # with open(path+'/data_tries.json') as data:
    #     tries = json.load(data)
    # with open(path+'/data_target.json') as data:
    #     target = json.load(data)
    # with open(path+'/data_grasp.json') as data:
    #     grasp = json.load(data)
    with open(path+'/'+modelname+'_data_tries.json') as data:
        tries = json.load(data)
    with open(path+'/'+modelname+'_data_target.json') as data:
        target = json.load(data)
    with open(path+'/'+modelname+'_data_grasp.json') as data:
        grasp = json.load(data)

    plot(path, tries, target, grasp, trials)
    write_summary(path, tries, target, grasp)


def plot_specific_model(path, tries, target, grasp, trials, modelname):
    succes_rate = dict.fromkeys(tries.keys())
    for obj in succes_rate.keys():
        t = tries[obj]
        if t == 0:
            t = 1
        acc_target = target[obj] / t
        acc_grasp = grasp[obj] / t
        succes_rate[obj] = (acc_target, acc_grasp)

    plt.rc('axes', titlesize=13)     # fontsize of the axes title
    plt.rc('axes', labelsize=12)    # fontsize of the x and y labels

    df = pd.DataFrame(succes_rate).T
    df.columns = ['Manipulation', 'Grasp']
    df = df.sort_values(by='Manipulation', ascending=True)
    ax = df.plot(kind='bar', color=['#88CCEE', '#CC6677'])
    # ax = df.plot(kind='bar', color=['b', 'r'])

    plt.xlabel('Object name')
    plt.ylabel('Succes rate (%)')
    plt.title(
        f'Succes rate for object grasping and manipulation | {trials} runs')
    plt.grid(color='#95a5a6', linestyle='-', linewidth=1, axis='y', alpha=0.5)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")
    plt.locator_params(axis="y", nbins=11)

    plt.legend(loc='lower right')
    plt.subplots_adjust(bottom=0.28)

    plt.savefig(path+'/'+modelname+'_plot.png')


def write_summary_specific_model(path, tries, target, grasp, modelname):
    with open(path+'/'+modelname+'_summary.txt', 'w') as f:
        total_tries = sum(tries.values())
        total_target = sum(target.values())
        total_grasp = sum(grasp.values())
        f.write('Total:\n')
        f.write(
            f'Target acc={total_target/total_tries:.3f} ({total_target}/{total_tries}) Grasp acc={total_grasp/total_tries:.3f} ({total_grasp}/{total_tries})\n')
        f.write('\n')
        f.write("Accuracy per object:\n")
        for obj in tries.keys():
            n_tries = tries[obj]
            n_t = target[obj]
            n_g = grasp[obj]
            f.write(
                f'{obj}: Target acc={n_t/n_tries:.3f} ({n_t}/{n_tries}) Grasp acc={n_g/n_tries:.3f} ({n_g}/{n_tries})\n')


def summarize_specific_model(path, trials, modelname):
    with open(path+'/'+modelname+'_data_tries.json') as data:
        tries = json.load(data)
    with open(path+'/'+modelname+'_data_target.json') as data:
        target = json.load(data)
    with open(path+'/'+modelname+'_data_grasp.json') as data:
        grasp = json.load(data)

    plot_specific_model(path, tries, target, grasp, trials, modelname)
    write_summary_specific_model(path, tries, target, grasp, modelname)
