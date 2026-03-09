# LiDAR-Based Online Control Barrier Function Synthesis for Safe Navigation in Unknown Environments
This repository provides a simulation reproducing the experiments from https://ieeexplore.ieee.org/abstract/document/10339852, including the simulator, source code, and data. 

## Experiment

You can view the experiment here:  
[NeboLab Experiment](https://www.youtube.com/watch?v=T_68wJbRJgY)

## Code and simulation assets for the paper:
> **LiDAR-Based Online Control Barrier Function Synthesis for Safe Navigation in Unknown Environments**  
> Shaghayegh Keyumarsi et al.

## Overview
The core idea is to learn a safety function online from LiDAR sensor data using a sparse Gaussian Process regression, then synthesize a Control Barrier Function that rectifies a nominal controller to guarantee collision-free navigation — without any prior knowledge of the environment.

### Key features of the approach:
> GP-based safety function learned online from LiDAR edge detections (Proposition 1)
> Single-pass data usage — no accumulation of past data across timesteps
> Handles arbitrary obstacle shapes and dynamic obstacles with one unified safety function
> CBF-QP minimally modifies the nominal controller to enforce safety 

### Simulation Results:
A video of the full simulation is included in the repository:
GP-CBF simulation_.mp4 shows all three robots navigating to their goals while avoiding static obstacles and each other using the GP-CBF safety controller. Additionally, the folder \lidar_gp_cbf\animation_result\sim2D_obstacle_GP contains per-robot GIF animations and plots generated after simulation, including:

Safety function color maps showing the GP prediction evolving in real time for each robot

Minimum LiDAR distance plots over time

Rectified control input plots (u_x, u_y, ‖u‖)

Safety function value h(t) over time

## Usage
-Run the simulation:
python sim2D_main.py
-This launches a 2D simulation with 3 unicycle robots navigating to their goals while avoiding static obstacles and each other. A matplotlib animation window will open showing the robots, the GP safety map, and minimum LiDAR distance plots in real time.

### Configuration
All tunable parameters are in sim2D_obstacle_GP.py:
-Scene parameters (SceneSetup)
-GP hyperparameters (SceneSetup)

### AnOutput
In sim2D_obstacle_GP.py, set:
### Outputs & animations
- item pythonSimSetup.save_animate = True.
- item Then run. The output GIF is saved to animation_result/sim2D_obstacle_GP/.



