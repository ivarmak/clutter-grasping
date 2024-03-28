# Target Driven Object Grasping in Highly Cluttered Scenarios through Domain Randomization and Active Segmentation 

<p align="center">
  <img src="img/isolate+grasp.gif" width="800" title="">
</p>

An example scenario of singulating the target TennisBall from a pile of objects, and a consecutive successful grasp.

For a full report on this master thesis project: https://fse.studenttheses.ub.rug.nl/28346/

## Requirements

Ensure you are running Python>=3.6.5 and import the required libraries by running:

```bash
cd ~
git clone https://github.com/ivarmak/clutter-grasping.git
cd ~/clutter-grasping
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

Furthermore, your .bashrc file needs to be edited.

```bash
cd ~
gedit .bashrc
```
and then add the following lines at the end of your .bashrc file

```sh
#This line is necessary for MoveIt! and Pybullet, otherwise the robot seems broken
export LC_NUMERIC="en_US.UTF-8"
```

close all your terminals and open one. 

## How to run experiments
We can perform a simulation experiment by running the 'simulation.py' script. As shown in the following image, we can perform experiments in four different grasping scenarios, including isolated, packed, piled, and cluttered scenarios:

<p align="center">
  <img src="img/scenarios.png" width="800" title="">
</p>


```bash
cd ~/clutter-grasping
python3 simulation.py clutter
```

  - Replace 'clutter' with 'iso', 'pack', or 'pile' to run one of the other scenarios.

  - Run 'simulation.py --help' to see a full list of options.
    
      - --runs=10 forces the system to run 10 experiments
      - In the ***environment/env.py*** file, we have provided a parameter namely ***SIMULATION_STEP_DELAY*** to control the speed of the simulator, this parameter should be tuned based on your hardware. 
       
      - After running the simulation, a summary of the results will be saved in the results folder as a pickled pandas dataframe. It can be opened by using pandas [read_pickle](https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html) function.

      - Furthermore, you can check the output of the grasping network by setting the ***--save-network-output=True***. The output will be saved into the ***network_output*** folder.

## Sources

This project uses adapted content from the following repositories:

Simulation Environment:
[Cognitive Robotics](https://github.com/SeyedHamidreza/cognitive_robotics_manipulation)

Object Detection:
[Mask RCNN](https://github.com/SriRamGovardhanam/wastedata-Mask_RCNN-multiple-classes) - [Original Implementation](https://github.com/matterport/Mask_RCNN)

Grasping:
[GR-ConvNet](https://github.com/skumra/robotic-grasping) - [GG-CNN](https://github.com/dougsm/ggcnn)

Used in training stage of Mask RCNN:
[ImgAug](https://github.com/aleju/imgaug) - [COCO API](https://github.com/cocodataset/cocoapi)
