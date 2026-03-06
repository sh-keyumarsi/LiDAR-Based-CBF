import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time


PYSIM = True
#PYSIM = False # for experiment or running via ROS

if PYSIM:
    from nebolab_experiment_setup import NebolabSetup
    from control_lib.goToGoal import Pcontrol
    from control_lib.cbf_single_integrator import cbf_si
    from control_lib.GP_h import GP
    from simulator.dynamics import Unicycle
    from simulator.plot_2D_unicycle import draw2DUnicyle
    from simulator.data_logger import dataLogger
    from simulator.detect_obstacle import DetectObstacle

else:
    from ..nebolab_experiment_setup import NebolabSetup
    from ..control_lib.goToGoal import Pcontrol
    from ..control_lib.cbf_single_integrator import cbf_si
    from ..control_lib.GP_h import GP
    from ..simulator.dynamics import Unicycle
    from ..simulator.plot_2D_unicycle import draw2DUnicyle
    from ..simulator.data_logger import dataLogger
    from ..simulator.detect_obstacle import DetectObstacle

# MAIN COMPUTATION
#------------------------------------------------------------------------------
class SceneSetup(): 
    # General variable needed to run the controller
    #number of robots
    robot_num = 3
    #robot_num = 1
    sense_dist=1 #lidar sensor range
    Pgain = 0.8 # for go-to-goal controller
    speed_limit = 0.2 # speed limit for the robot,
    gamma_staticObs = 2000
    #init_pos = np.array([[-1.2,0, 0]]) 
    #init_theta = np.array([[0]])
    #goal_pos = np.array([[1.2, 0,0]])
    init_pos = np.array([[-1.8,  0.6, 0],[-1.5,  -0.5, 0] ,[ 1.5,  0.6, 0]]) 
    init_theta = np.array([[0],[0],[-np.pi]])
    goal_pos = np.array([ [0.5,  -1., 0],[ 0.3,  1.2, 0],[-0.5,  -1.3, 0]])
    # Here we assume any final pose is OK
    # Define Obstacles
    # obstacle = []
    obstacle = [np.array([[-1.15, 1.5 ,0], [-1.15, 0.9 ,0], [-1, 0.9 ,0], [-1, 1.5 ,0], [-1.15, 1.5 ,0],[-1.15, 1.5 ,0]]),
                   np.array([[-1.15, -1.5 ,0], [-1.15, -0.9 ,0], [-1, -0.9 ,0], [-1, -1.5 ,0], [-1.15, -1.5 ,0],[-1.15, -1.5 ,0]]),
                   0.7*np.array([[-0.5, -0.5 ,0], [-0.5, 0.5 ,0], [1.2, 0.5 ,0], [1.2,-0.5 ,0], [-0.5, -0.5 ,0]]),
                   np.array([[-1.85,0.1 ,0], [-2.6,0.1  ,0], [-2.6,-0.05 ,0], [-1.85, -0.05 ,0], [-1.85,0.1 ,0]]),
                  #0.4*np.array([[ 1., 0., 0], [ 0.5, np.sqrt(3)/2, 0], [-0.5, np.sqrt(3)/2, 0], [-1., 0., 0], [-0.5, -np.sqrt(3)/2, 0], [ 0.5, -np.sqrt(3)/2, 0], [ 1.,  0., 0]])
                  ]       
   
    '''it is assumed that the first position is safe with no obstacles in the lidar sensing range'''
    # Default values for sensing data --> assume with largest possible number with 1 deg resolution
    default_range_data = sense_dist*np.ones((robot_num, 360))
    
    '''................GP Controller params.................................'''
    min_d_sample = 0.05 # Define the minimum distance GP sample
    #GP hyper parameters: l, sigma_f, sigma_y
    hypers_gp=np.array([[0.17,1,1e-2]]).T
    #grid size for plotting the color map of the safety function
    grid_size_plot= 0.2
    #the upper limit dh/dt for the dynamic environment to be applied in CBF
    #gamma(h)-dh/dt
    dh_dt=0.4

# General class for computing the controller input
class Controller():
    def __init__(self): # INITIALIZE CONTROLLER
        self.cbf = [cbf_si() for _ in range(SceneSetup.robot_num)] # For collision avoidance        
        self.gp = [GP(SceneSetup.hypers_gp, SceneSetup.min_d_sample, SceneSetup.grid_size_plot, SceneSetup.dh_dt) for _ in range(SceneSetup.robot_num)]
        
    def compute_control(self, feedback, computed_control): # INITIALIZE CONTROLLER

        # Reset monitor properties
        computed_control.reset_monitor()
    
        #input("Press Enter to continue...")

        for i in range(SceneSetup.robot_num):
            #print('robot',i)
            # Collect States
            start_time = time.time() #to measure computation time
            # ----------------------------------------------------------------
            current_q = feedback.get_lahead_i_pos(i) #look ahead position for Go to Goal 
            current_q_center=feedback.get_robot_i_pos(i) #accurate position
            goal = SceneSetup.goal_pos[i]
            sensing_data = feedback.get_robot_i_range_data(i)  # Get range data       
            sensor_pos_data = feedback.get_robot_i_detected_pos(i) # Get pos of range data 
            current_data_X=np.reshape(current_q[0:2], (1,2))
            #current_data_X_center=np.reshape(current_q_center[0:2], (1,2))
 
            #------------------------------------------------------------------
            # Implementation of Control
            # Calculate nominal controller
            u_nom = Pcontrol(current_q, SceneSetup.Pgain, goal)[0:2]

            if SceneSetup.speed_limit > 0.:
                # set speed limit
                norm = np.hypot(u_nom[0], u_nom[1])
                if norm > SceneSetup.speed_limit: u_nom = (SceneSetup.speed_limit-1e-7)* u_nom / norm # max 
                #print('u_nom', u_nom)
            '''___________________process the range data into GP____________''' 
            #Update GP
            self.gp[i].new_iter()
            #Sparsely Sample detected edges: check if any obstacle is detected and collect with sampling distance min_dis constraint
            for j in range(360):
                if sensing_data[j] < SceneSetup.sense_dist: 
                    #get the location of the detected obstacle edges 
                    edge_data_X=np.reshape(sensor_pos_data[j,0:2], (1,2))
                    #edge_data_Y=np.array([[-2]])
                    self.gp[i].set_new_data(edge_data_X)
   
            # ------------------------------------------------
                
            # for differentiability of the GP model data set must be non empty
            if self.gp[i].N == 0:
                u=u_nom
                h = np.array([[1]])
                true_gp_h=h
                norm_u_nom = np.hypot(u_nom[0], u_nom[1])
                #if norm_u_nom > SceneSetup.speed_limit: print('error: u_nom is larger than speed limit', norm_u_nom, SceneSetup.speed_limit)
                #print('No data collected yet, using nominal controller', u_nom)
            else:
                # Construct CBF setup #remove previous runs CBF parametrs
                self.cbf[i].reset_cbf()
                '''____________________Compute Lidar-GP_CBF_________________''' 
                gp_G, gp_h, true_gp_h =self.gp[i].get_cbf_safety_prediction(current_data_X) 
                if true_gp_h<0:
                    print('the safety function is negative! increasec dh/dt in CBF constraint', 'h', true_gp_h,'d', min(sensing_data) )
                # Add GP-CF constraint
                self.cbf[i].add_computed_constraint(gp_G, gp_h)

                '''____________________CBF OPT______________________________'''
                # Ensure safety by rectifying the control input using CBF
                u = self.cbf[i].compute_safe_controller(u_nom, SceneSetup.speed_limit/np.sqrt(2))

                
                #print('d',min(sensing_data), 'h', true_gp_h[0,0])
                #print('u', u, 'u_nom', u_nom)

            end_time = time.time()
            #print('pos',current_data_X,'min dis to obs',min(sensing_data))
            # print('||u_rec||', "{:.2e}".format(np.linalg.norm(u)),'||u_rec||/||u_nom||', "{:.2e}".format(np.linalg.norm(u)/np.linalg.norm(u_nom)))
            
            # SAVING the data_X, data_Y, N, and ... for post processing
            computed_control.save_monitored_info( "data_X_"+str(i), self.gp[i].data_X )
            computed_control.save_monitored_info( "data_Y_"+str(i), self.gp[i].data_Y )
            computed_control.save_monitored_info( "data_N_"+str(i), self.gp[i].N )
            computed_control.save_monitored_info( "data_k_"+str(i), self.gp[i].k )
            #position of sys
            computed_control.save_monitored_info( "posc_x_"+str(i), current_q_center[0] )
            computed_control.save_monitored_info( "posc_y_"+str(i), current_q_center[1] )
            # store GP h function and sensor's minimum readings (min distance to obstacle)
            computed_control.save_monitored_info( "h_gp_"+str(i), true_gp_h[0,0] )
            computed_control.save_monitored_info( "min_lidar_"+str(i), min(SceneSetup.sense_dist,np.min(sensing_data)) )
            # Store computation time
            computed_control.save_monitored_info( "run_time_"+str(i), end_time - start_time )
            # ------------------------------------------------
            computed_control.set_i_vel_xy(i, u[:2])
            # storing the rectified input
            computed_control.save_monitored_info( "u_x_"+str(i), u[0] )
            computed_control.save_monitored_info( "u_y_"+str(i), u[1] )
            computed_control.save_monitored_info( "u_norm_"+str(i), np.linalg.norm(u) )
            # store information to be monitored/plot
            computed_control.save_monitored_info( "u_nom_x_"+str(i), u_nom[0] )
            computed_control.save_monitored_info( "u_nom_y_"+str(i), u_nom[1] )
            computed_control.save_monitored_info( "pos_x_"+str(i), current_q[0] )
            computed_control.save_monitored_info( "pos_y_"+str(i), current_q[1] )

        computed_control.pass_gp_classes(self.gp)


#-----------------------------------------
# CLASS FOR CONTROLLER'S INPUT AND OUTPUT
#-----------------------------------------
class ControlOutput():
    # Encapsulate the control command to be passed 
    # from controller into sim/experiment
    def __init__(self):
        # Initialize the formation array
        self.__all_velocity_input_xyz = np.zeros([SceneSetup.robot_num, 3])
        # introduce new variable to store H matrix
        self.__all_H_matrix = {}
        self.__recorded_gp_classes = None
    
    def get_all_vel_xy(self): return self.__all_velocity_input_xyz[:,:2]

    def get_i_vel_xy(self, ID): return self.__all_velocity_input_xyz[ID,:2]
    def set_i_vel_xy(self, ID, input_xy):
        self.__all_velocity_input_xyz[ID,:2] = input_xy

    # Allow the options to monitor state / variables over time
    def reset_monitor(self): self.__monitored_signal = {}
    def save_monitored_info(self, label, value): 
        # NOTE: by default name the label with the index being the last
        # example p_x_0, p_y_0, h_form_1_2, etc.
        self.__monitored_signal[label] = value
    # Allow retrieval from sim or experiment
    def get_all_monitored_info(self): return self.__monitored_signal

    # SPECIFIC ADDITIONAL FOR PLOTTING GP
    def pass_gp_classes(self, gp_classes): self.__recorded_gp_classes = gp_classes
    def get_gp_classes(self): return self.__recorded_gp_classes
    


class FeedbackInformation():
    # Encapsulate the feedback information to be passed 
    # from sim/experiment into controller
    def __init__(self):
        # Set the value based on initial values
        self.set_feedback(SceneSetup.init_pos, SceneSetup.init_theta)
        # Set the range data and detected pos
        n, m = SceneSetup.default_range_data.shape
        self.__all_detected_pos = np.zeros((n, m, 3))
        self.__sensing_linspace = np.linspace(0., 2*np.pi, num=m, endpoint=False)
        self.set_sensor_reading(SceneSetup.default_range_data)

    # To be assigned from the SIM or EXP side of computation
    def set_feedback(self, all_robots_pos, all_robots_theta, all_lahead_pos=None):
        # update all robots position and theta
        self.__all_robot_pos = all_robots_pos.copy()
        self.__all_robot_theta = all_robots_theta.copy()
        self.__all_lahead_pos = all_robots_pos.copy() # temporary
        if all_lahead_pos is not None:
            self.__all_lahead_pos = all_lahead_pos.copy()
        else: 
            # update lookahead position for each robot
            for i in range(SceneSetup.robot_num):
                th = all_robots_theta[i]
                ell_si = np.array([np.cos(th), np.sin(th), 0], dtype=object) * NebolabSetup.TB_L_SI2UNI
                self.__all_lahead_pos[i,:] = all_robots_pos[i,:] + ell_si
                
    def set_sensor_reading(self, all_range_data): 
        self.__all_range_data = all_range_data.copy()
        # update the detected position for each robot
        for i in range( all_range_data.shape[0] ):
            sensing_angle_rad = self.__all_robot_theta[i] + self.__sensing_linspace
            self.__all_detected_pos[i,:,0] = self.__all_robot_pos[i,0] + all_range_data[i]* np.cos( sensing_angle_rad )
            self.__all_detected_pos[i,:,1] = self.__all_robot_pos[i,1] + all_range_data[i]* np.sin( sensing_angle_rad )


    # To allow access from the controller computation
    def get_robot_i_pos(self, i):   return self.__all_robot_pos[i,:]
    def get_robot_i_theta(self, i): return self.__all_robot_theta[i]
    def get_lahead_i_pos(self, i):  return self.__all_lahead_pos[i,:]

    # get all robots information
    def get_all_robot_pos(self):   return self.__all_robot_pos
    def get_all_robot_theta(self): return self.__all_robot_theta
    def get_all_lahead_pos(self):  return self.__all_lahead_pos

    # get range data
    def get_robot_i_detected_pos(self, i): return self.__all_detected_pos[i]
    def get_robot_i_range_data(self, i): return self.__all_range_data[i,:]


# ONLY USED IN SIMULATION
#-----------------------------------------------------------------------
class SimSetup():

    Ts = 0.02 # in second. Determine Visualization and dynamic update speed
    tmax =25 # simulation duration in seconds (only works when save_animate = True)
    save_animate = False # True: saving but not showing, False: showing animation but not real time
    save_data = True # log data using pickle
    plot_saved_data = True

    sim_defname = 'animation_result/sim2D_obstacle_GP/sim_'
    sim_fname_output = r''+sim_defname+'.gif'
    trajectory_trail_lenTime = tmax # Show all trajectory
    sim_fdata_log = sim_defname + '_data.pkl'    

    timeseries_window = tmax #5 # in seconds, for the time series data

    robot_angle_bound = np.append( np.linspace(0., 2*np.pi, num=8, endpoint=False), 0 ) + np.pi/8
    robot_rad = 0.1


# General class for drawing the plots in simulation
class SimulationCanvas():
    def __init__(self):
        # self.__sim_ctr = 0
        # self.__max_ctr = round(SimSetup.tmax / SimSetup.Ts)
        self.__cur_time = 0.

        # Initiate the robot
        self.__robot_dyn = [None]*SceneSetup.robot_num
        for i in range(SceneSetup.robot_num):
            self.__robot_dyn[i] = Unicycle(SimSetup.Ts, SceneSetup.init_pos[i], SceneSetup.init_theta[i], ell = NebolabSetup.TB_L_SI2UNI)

        # Initiate ranging sensors for the obstacles
        self.__rangesens = DetectObstacle( detect_max_dist=SceneSetup.sense_dist, angle_res_rad=np.pi/180)
        for i in range(len(SceneSetup.obstacle)):
            self.__rangesens.register_obstacle_bounded( 'obs'+str(i), SceneSetup.obstacle[i] )

        # Initiate data_logger
        # self.log = dataLogger( self.__max_ctr )
        self.log = dataLogger(round(SimSetup.tmax / SimSetup.Ts) + 1)
        # Initiate the plotting
        self.__initiate_plot()

        # flag to check if simulation is still running
        self.is_running = True


    def update_simulation(self, control_input, feedback):
        if self.__cur_time < SimSetup.tmax:
            # Store data to log
            self.log.store_dictionary( control_input.get_all_monitored_info() )
            self.log.time_stamp( self.__cur_time )

            self.__cur_time += SimSetup.Ts
            # Set array to be filled
            all_robots_pos = np.zeros( SceneSetup.init_pos.shape )
            all_robots_theta = np.zeros( SceneSetup.init_theta.shape )
            all_range_data = SceneSetup.default_range_data.copy()
            # IMPORTANT: advance the robot's dynamic, and update feedback information
            for i in range(SceneSetup.robot_num):
                self.__robot_dyn[i].set_input(control_input.get_i_vel_xy(i), "u")
                state = self.__robot_dyn[i].step_dynamics() 

                all_robots_pos[i,:2] = state['q'][:2]
                all_robots_theta[i] = state['theta']

                # Update robot shape to be used for range detection
                v_angles = SimSetup.robot_angle_bound + all_robots_theta[i]
                robot_shape = np.array([ np.cos(v_angles), np.sin(v_angles), v_angles*0])*SimSetup.robot_rad
                robot_bounds = np.transpose(robot_shape + all_robots_pos[i,:3].reshape(3,1))
                self.__rangesens.register_obstacle_bounded(i, robot_bounds)

            for i in range(SceneSetup.robot_num):
                # update sensor data by excluding its own
                all_range_data[i,:] = self.__rangesens.get_sensing_data(
                    all_robots_pos[i,0], all_robots_pos[i,1], all_robots_theta[i], exclude=[i] )

            # UPDATE FEEDBACK for the controller
            feedback.set_feedback(all_robots_pos, all_robots_theta)
            feedback.set_sensor_reading(all_range_data)

        else: # No further update
            if self.is_running:
                if SimSetup.save_data: 
                    self.log.save_to_pkl( SimSetup.sim_fdata_log )

                    if SimSetup.plot_saved_data: 
                        from scenarios.obstacle_GP_pickleplot import scenario_pkl_plot
                        scenario_pkl_plot()

                print( f"Stopping the simulation, tmax reached: {self.__cur_time:.2f} s" )
                # if not SimSetup.save_animate: exit() # force exit
                self.is_running = False 
            # else: # Do nothing
            
        # Update plot
        self.__update_plot( feedback, control_input)


    # PROCEDURES RELATED TO PLOTTING - depending on the scenarios
    #---------------------------------------------------------------------------------
    def __initiate_plot(self):
        # Initiate the plotting
        # For now plot 2D with 2x2 grid space, to allow additional plot later on
        rowNum, colNum = 4, 4 
        self.fig = plt.figure(figsize=(4*colNum, 3*rowNum), dpi= 100)
        gs = GridSpec( rowNum, colNum, figure=self.fig)

        # MAIN 2D PLOT FOR UNICYCLE ROBOTS
        # ------------------------------------------------------------------------------------
        ax_2D = self.fig.add_subplot(gs[0:2,0:2]) # Always on
        # Only show past several seconds trajectory
        tail_len = int(SimSetup.trajectory_trail_lenTime/SimSetup.Ts) 
        trajTail_datanum = [tail_len, tail_len, tail_len, tail_len]

        self.__drawn_2D = draw2DUnicyle( ax_2D, SceneSetup.init_pos, SceneSetup.init_theta,
            field_x = NebolabSetup.FIELD_X, field_y = NebolabSetup.FIELD_Y, pos_trail_nums=trajTail_datanum )

        # Draw goals and obstacles
        for i in [0]:#range(SceneSetup.robot_num):
            ax_2D.add_patch( plt.Circle( (SceneSetup.goal_pos[i][0], SceneSetup.goal_pos[i][1]), 0.03, color='g' ) )
        for obs in SceneSetup.obstacle:
            ax_2D.plot(obs[:,0], obs[:,1], 'k')

        # Display simulation time
        self.__drawn_time = ax_2D.text(0.78, 0.99, 't = 0 s', color = 'k', fontsize='large', 
            horizontalalignment='left', verticalalignment='top', transform = ax_2D.transAxes)

        # Display sensing data
        self.__pl_sens = {}
        __colorList = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(SceneSetup.robot_num):
            self.__pl_sens[i], = ax_2D.plot(0, 0, '.', color=__colorList[i])

        # ADDITIONAL PLOT
        # ------------------------------------------------------------------------------------
        # Plot nominal velocity in x- and y-axis
        # self.__ax_unomx = self.fig.add_subplot(gs[2,0])
        # self.__ax_unomy = self.fig.add_subplot(gs[2,1])
        # self.log.plot_time_series_batch( self.__ax_unomx, 'u_nom_x_' ) 
        # self.log.plot_time_series_batch( self.__ax_unomy, 'u_nom_y_' ) 
        # # Plot position in x- and y-axis
        # self.__ax_pos_x = self.fig.add_subplot(gs[2,0])
        # self.__ax_pos_y = self.fig.add_subplot(gs[2,1])
        # self.log.plot_time_series_batch( self.__ax_pos_x, 'pos_x_' ) 
        # self.log.plot_time_series_batch( self.__ax_pos_y, 'pos_y_' ) 

        #Plot GP H
        self.__ax_gp = {}
        self.__ax_gp[0] = self.fig.add_subplot(gs[0:2,2:4])
        self.__ax_gp[0].set(xlabel="x [m]", ylabel="y [m]")
        self.__ax_gp[0].set(xlim=(NebolabSetup.FIELD_X[0]-0.1, NebolabSetup.FIELD_X[1]+0.1))
        self.__ax_gp[0].set(ylim=(NebolabSetup.FIELD_Y[0]-0.1, NebolabSetup.FIELD_Y[1]+0.1))
        self.__ax_gp[0].set_aspect('equal', adjustable='box', anchor='C')        
        for obs in SceneSetup.obstacle:
            self.__ax_gp[0].plot(obs[:,0], obs[:,1], 'k')
        # self.log.plot_time_series_batch( self.__ax_unomy, 'u_nom_y_' ) 
        # self.__ax_gp[1] = self.fig.add_subplot(gs[0,3])
        # self.__ax_gp[2] = self.fig.add_subplot(gs[1,2])
        # self.__ax_gp[3] = self.fig.add_subplot(gs[1,3])

        self.__gp_pl_trail = [None]*SceneSetup.robot_num
        self.__gp_pl_pos = [None]*SceneSetup.robot_num
        # # Plot the first trajectory trail
        for i in [0]:#range(SceneSetup.robot_num):
            trail_data_i = self.__drawn_2D.extract_robot_i_trajectory(i)
            self.__gp_pl_trail[i], = self.__ax_gp[0].plot(trail_data_i[:,0], trail_data_i[:,1], 
                '--', color=__colorList[i])
            self.__gp_pl_pos[i], = self.__ax_gp[0].plot(trail_data_i[0,0], trail_data_i[0,1], 
                'x', color=__colorList[i])

        # Plot minimum LIDAR readings & h values from GP-CBF 
        self.__ax_min_lidar0 = self.fig.add_subplot(gs[2,0])
        self.__ax_min_lidar0.set(xlabel="t [s]", ylabel="Min LiDAR dis (Blue Bot) [m]")
        self.__ax_min_lidar0.set(ylim=(-0.1, SceneSetup.sense_dist+0.1))
        self.__ax_min_lidar0.set(xlim=(-0.1, SimSetup.tmax + 0.1))
        self.__ax_min_lidar0.grid(True)
        self.__pl_min_lidar0 = {}
        for i in [0]:#range(SceneSetup.robot_num):
            self.__pl_min_lidar0[i], = self.__ax_min_lidar0.plot(0, 0, '-', color=__colorList[i])


        # Plot minimum LIDAR readings & h values from GP-CBF 
        self.__ax_min_lidar1 = self.fig.add_subplot(gs[2,1])
        self.__ax_min_lidar1.set(xlabel="t [s]", ylabel="Min LiDAR dis (Orange Bot) [m]")
        self.__ax_min_lidar1.set(ylim=(-0.1, SceneSetup.sense_dist+0.1))
        self.__ax_min_lidar1.set(xlim=(-0.1, SimSetup.tmax + 0.1))
        self.__ax_min_lidar1.grid(True)
        self.__pl_min_lidar1 = {}
        for i in [1]:#range(SceneSetup.robot_num):
            self.__pl_min_lidar1[i], = self.__ax_min_lidar1.plot(0, 0, '-', color=__colorList[i])

        # Plot minimum LIDAR readings & h values from GP-CBF 
        self.__ax_min_lidar2 = self.fig.add_subplot(gs[2,2])
        self.__ax_min_lidar2.set(xlabel="t [s]", ylabel="Min LiDAR dis (Green Bot) [m]")
        self.__ax_min_lidar2.set(ylim=(-0.1, SceneSetup.sense_dist+0.1))
        self.__ax_min_lidar2.set(xlim=(-0.1, SimSetup.tmax + 0.1))
        self.__ax_min_lidar2.grid(True)
        self.__pl_min_lidar2 = {}
        for i in [2]:#range(SceneSetup.robot_num):
            self.__pl_min_lidar2[i], = self.__ax_min_lidar2.plot(0, 0, '-', color=__colorList[i])

            
        # self.__ax_u_cbf = self.fig.add_subplot(gs[2,1])
        # self.__ax_u_cbf.set(xlabel="t [s]", ylabel=r'$\|u\|$')
        # self.__ax_u_cbf.set(ylim=(-1.1*SceneSetup.speed_limit, 1.1*SceneSetup.speed_limit))
        # self.__ax_u_cbf.set(xlim=(-0.1, SimSetup.tmax + 0.1))
        # self.__ax_u_cbf.grid(True)
        # self.__pl_u_cbf = {}
        # for i in [0]:#range(SceneSetup.robot_num):
        #     self.__pl_u_cbf[i], = self.__ax_u_cbf.plot(0, 0, '-', color=__colorList[i])
            
        # self.__ax_u1_cbf = self.fig.add_subplot(gs[2,2])
        # self.__ax_u1_cbf.set(xlabel="t [s]", ylabel='$u_x$')
        # self.__ax_u1_cbf.set(ylim=(-1.1*SceneSetup.speed_limit, 1.1*SceneSetup.speed_limit))
        # self.__ax_u1_cbf.set(xlim=(-0.1, SimSetup.tmax + 0.1))
        # self.__ax_u1_cbf.grid(True)
        # self.__pl_u1_cbf = {}
        # for i in [0]:#range(SceneSetup.robot_num):
        #     self.__pl_u1_cbf[i], = self.__ax_u1_cbf.plot(0, 0, '-', color=__colorList[i])
            
        # self.__ax_u2_cbf = self.fig.add_subplot(gs[2,3])
        # self.__ax_u2_cbf.set(xlabel="t [s]", ylabel='$u_y$')
        # self.__ax_u2_cbf.set(ylim=(-1.1*SceneSetup.speed_limit, 1.1*SceneSetup.speed_limit))
        # self.__ax_u2_cbf.set(xlim=(-0.1, SimSetup.tmax + 0.1))
        # self.__ax_u2_cbf.grid(True)
        # self.__pl_u2_cbf = {}
        # for i in [0]:#range(SceneSetup.robot_num):
        #     self.__pl_u2_cbf[i], = self.__ax_u2_cbf.plot(0, 0, '-', color=__colorList[i])
        
        #to plot safety function value
        # self.__ax_gp_cbf = self.fig.add_subplot(gs[2,2:4])
        # self.__ax_gp_cbf.set(xlabel="t [s]", ylabel="h")
        # self.__ax_gp_cbf.set(ylim=(-0.1, 1.1))
        # self.__ax_gp_cbf.set(xlim=(-0.1, SimSetup.tmax + 0.1))
        # self.__ax_gp_cbf.grid(True)
        # self.__pl_gp_cbf = {}
        # for i in [0]:#range(SceneSetup.robot_num):
        #     self.__pl_gp_cbf[i], = self.__ax_gp_cbf.plot(0, 0, '-', color=__colorList[i])

        plt.tight_layout()


    def __update_plot(self, feedback, control_input):
        # UPDATE 2D Plotting: Formation and Robots
        
        self.__drawn_2D.update( feedback.get_all_robot_pos(), feedback.get_all_robot_theta() )
        self.__drawn_time.set_text('t = '+f"{self.__cur_time:.1f}"+' s')

        # update display of sensing data
        for i in [0]:#range(SceneSetup.robot_num):
            sensed_pos = feedback.get_robot_i_detected_pos(i)
            self.__pl_sens[i].set_data(sensed_pos[:,0], sensed_pos[:,1])
        # if self.gp[i].add_edge :
        #     self.gp[i].sensed_pos=sensed_pos[self.gp[i].sensed_edge,0]

        __colorList = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # update GP plot
        all_gp_class = control_input.get_gp_classes()
        for i in [0]: #range(SceneSetup.robot_num):
            all_gp_class[i].draw_gp_whole_map_prediction( 
                self.__ax_gp[0], NebolabSetup.FIELD_X, NebolabSetup.FIELD_Y, i, feedback.get_robot_i_pos(i), SceneSetup.sense_dist, color=__colorList[i])
            # Update trajectory trail
            trail_data_i = self.__drawn_2D.extract_robot_i_trajectory(i)
            self.__gp_pl_trail[i].set_data(trail_data_i[:,0], trail_data_i[:,1])
            self.__gp_pl_pos[i].set_data([trail_data_i[0,0]],[ trail_data_i[0,1]])

        # get data from Log
        log_data, max_idx = self.log.get_all_data()
        min_idx = 0
        # Setup for moving window horizon
        # if self.__cur_time < SimSetup.timeseries_window:
        #     t_range = (-0.1, SimSetup.timeseries_window + 0.1)
        #     min_idx = 0
        # else:
        #     t_range = (self.__cur_time - (SimSetup.timeseries_window + 0.1), self.__cur_time + 0.1)
        #     min_idx = max_idx - round(SimSetup.timeseries_window / SimSetup.Ts)

        # update minimum LIDAR readings & h values from GP-CBF 
        for i in [0]:#range(SceneSetup.robot_num):
            time_span = log_data['time'][min_idx:max_idx]
            min_lidar_val0 = log_data["min_lidar_" + str(i)][min_idx:max_idx]
            min_lidar_val1 = log_data["min_lidar_" + str(i+1)][min_idx:max_idx]
            min_lidar_val2 = log_data["min_lidar_" + str(i+2)][min_idx:max_idx]
            # u_val = log_data["u_norm_" + str(i)][min_idx:max_idx]
            # u1_val = log_data["u_nom_x_" + str(i)][min_idx:max_idx]
            # u2_val = log_data["u_nom_y_" + str(i)][min_idx:max_idx]

            self.__pl_min_lidar0[i].set_data(time_span, min_lidar_val0)
            self.__pl_min_lidar1[i+1].set_data(time_span, min_lidar_val1)
            self.__pl_min_lidar2[i+2].set_data(time_span, min_lidar_val2)
            # self.__pl_u_cbf[i].set_data(time_span, u_val)
            # self.__pl_u1_cbf[i].set_data(time_span, u1_val)
            # self.__pl_u2_cbf[i].set_data(time_span, u2_val)
            
            
        # Move the time-series window
        # self.__ax_min_lidar.set(xlim=t_range)
        # self.__ax_gp_cbf.set(xlim=t_range)

        # update nominal velocity in x- and y-axis
        # self.log.update_time_series_batch( 'u_nom_x_', data_minmax=(min_idx, max_idx)) 
        # self.log.update_time_series_batch( 'u_nom_y_', data_minmax=(min_idx, max_idx)) 

        # # update position in x- and y-axis
        # self.log.update_time_series_batch( 'pos_x_', data_minmax=(min_idx, max_idx)) 
        # self.log.update_time_series_batch( 'pos_y_', data_minmax=(min_idx, max_idx)) 



# ONLY USED IN EXPERIMENT
#-----------------------------------------------------------------------
class ExpSetup():
    parent_fold = '' #'/home/localadmin/ros_ws/src/lidar_gp_cbf/lidar_gp_cbf/experiment_result/'
    exp_defname = parent_fold + 'ROSTB_LIDAR_GP_CBF'
    exp_fdata_log = exp_defname + '_data.pkl'
    ROS_RATE = 20
    LiDAR_RATE = 5
    log_duration = 90
    ROS_NODE_NAME = 'ROSTB_LIDAR_GP_CBF'

class ExperimentEnv():
    def __init__(self):
        self.global_lahead = [None for _ in range(SceneSetup.robot_num)]
        self.global_poses = [None for _ in range(SceneSetup.robot_num)]
        # self.scan_LIDAR = [None for _ in range(SceneSetup.robot_num)]
        self.scan_LIDAR = SceneSetup.default_range_data.copy()
        # Initiate data_logger
        self.__cur_time = 0.
        self.log = dataLogger( ExpSetup.log_duration * ExpSetup.ROS_RATE )

    # NOTES: it seems cleaner to do it this way
    # rather than dynamically creating the callbacks
    def pos_callback(self, msg, index): self.global_lahead[index] = msg
    def posc_callback(self, msg, index): self.global_poses[index] = msg
    def scan_LIDAR_callback(self, msg, index): self.scan_LIDAR[index, :] = np.array(msg.ranges)

    def update_feedback(self, feedback):
        all_robots_pos = np.zeros([SceneSetup.robot_num, 3])
        all_robots_theta = np.zeros([SceneSetup.robot_num, 1])
        all_robots_pos_ahead = np.zeros([SceneSetup.robot_num, 3])
        # TODO: below might not work for robots less than 4
        for i in range(SceneSetup.robot_num):
            all_robots_pos[i,0] = self.global_poses[i].x
            all_robots_pos[i,1] = self.global_poses[i].y
            all_robots_theta[i] = self.global_poses[i].theta
            all_robots_pos_ahead[i,0] = self.global_lahead[i].x
            all_robots_pos_ahead[i,1] = self.global_lahead[i].y
        # UPDATE FEEDBACK for the controller
        feedback.set_feedback(all_robots_pos, all_robots_theta, all_robots_pos_ahead)

        # if self.__cur_time % ExpSetup.LiDAR_rate < SimSetup.Ts:
        feedback.set_sensor_reading(self.scan_LIDAR)


    def get_i_vlin_omega(self, i, control_input):
        # Inverse Look up ahead Mapping (u_z remain 0.)
        #   V = u_x cos(theta) + u_y sin(theta)
        #   omg = (- u_x sin(theta) + u_y cos(theta)) / l
        u = control_input.get_i_vel_xy(i)
        theta = self.global_poses[i].theta
        vel_lin = u[0]*np.cos(theta) + u[1]*np.sin(theta)
        vel_ang = (- u[0]*np.sin(theta) + u[1]*np.cos(theta))/NebolabSetup.TB_L_SI2UNI
        return vel_lin, vel_ang
        # return 0, 0

    def update_log(self, control_input):
        # Store data to log
        self.log.store_dictionary( control_input.get_all_monitored_info() )
        self.log.time_stamp( self.__cur_time )
        # NOT REAL TIME FOR NOW. TODO: fix this with real time if possible
        self.__cur_time += 1./ExpSetup.ROS_RATE

    def save_log_data(self): 
        self.log.save_to_pkl( ExpSetup.exp_fdata_log )
        # automatic plot if desired
        # if SimSetup.plot_saved_data: 
        #     from scenarios.Resilient_pickleplot import scenario_pkl_plot
        #     # Quick fix for now --> TODO: fix this
        #     SimSetup.sim_defname = ExpSetup.exp_defname
        #     SimSetup.sim_fdata_log = ExpSetup.exp_fdata_log
        #     # Plot the pickled data
        #     scenario_pkl_plot()