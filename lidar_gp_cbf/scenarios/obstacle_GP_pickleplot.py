import pickle
import numpy as np
import matplotlib.pyplot as plt

from scenarios.sim2D_obstacle_GP import SceneSetup, SimSetup, ExpSetup

def preamble_setting():  # when manual plotting is needed
    # Temporary fix for animation or experiment data TODO: fix this later
    SimSetup.sim_defname = 'animation_result/sim2D_obstacle_GP/sim_'
    #SimSetup.sim_defname = 'experiment_result/ROSTB_LIDAR_GP_CBF'
    SimSetup.sim_fdata_log = SimSetup.sim_defname + '_data.pkl'


def plot_pickle_log_time_series_batch_keys( ax, datalog_data, __end_idx, pre_string ):
    # check all matching keystring
    time_data = datalog_data['time'][:__end_idx]
    matches = [key for key in datalog_data if key.startswith(pre_string)]
    data_min, data_max = 0., 0.
    for key in matches:
        key_data = datalog_data[key][:__end_idx]
        ax.plot(time_data, key_data, label=key.strip(pre_string))
        # update min max for plotting
        data_min = min( data_min, min(i for i in key_data if i is not None) ) 
        data_max = max( data_max, max(i for i in key_data if i is not None) ) 
    # adjust time window
    ax.grid(True)
    ax.set(xlim= (time_data[0]-0.1, time_data[-1]+0.1), 
        ylim= (data_min-0.1, data_max+0.1))

def plot_pickle_log_time_series_batch_robotid( ax, datalog_data, __end_idx, pre_string, id_name=None ):
    # check all matching keystring
    time_data = datalog_data['time'][:__end_idx]
    data_min, data_max = 0., 0.
    if id_name is None: id_name = [str(i) for i in range(SceneSetup.robot_num)]
    for i in range(SceneSetup.robot_num):
        key = pre_string+str(i)
        key_data = datalog_data[key][:__end_idx]
        ax.plot(time_data, key_data, color=SceneSetup.robot_color[i], label=id_name[i])
        # update min max for plotting
        data_min = min( data_min, min(i for i in key_data if i is not None) ) 
        data_max = max( data_max, max(i for i in key_data if i is not None) ) 
    # adjust time window
    ax.grid(True)
    ax.set(xlim= (time_data[0]-0.1, time_data[-1]+0.1), 
        ylim= (data_min-0.1, data_max+0.1))

def plot_pickle_robot_distance( ax, datalog_data, __end_idx, pre_pos_x, pre_pos_y ):
    dist_max = 0.
    time_data = datalog_data['time'][:__end_idx]
    for i in range(SceneSetup.robot_num):
        for j in range(SceneSetup.robot_num):
            if (i < j):
                dist = [np.sqrt( 
                    ( datalog_data[pre_pos_x+str(i)][k] - datalog_data[pre_pos_x+str(j)][k] )**2 + 
                    ( datalog_data[pre_pos_y+str(i)][k] - datalog_data[pre_pos_y+str(j)][k] )**2 )
                    for k in range(__end_idx) ]
                dist_max = max( dist_max, max(dist) ) 
                # update line plot
                ax.plot(time_data, dist, label='$i={},\ j={}$'.format(i+1,j+1))
    # Plot the additional steady line at y = 0.2 in red
    ax.plot(time_data, [0.105] * len(time_data), color='red', linestyle='-', label='y=0.105')
    # update axis length
    ax.grid(True)
    ax.set(xlim= (time_data[0]-0.1, time_data[-1]+0.1),
        ylim= (-0.1, dist_max+0.1) )


def plot_pickle_individual_id( ax, datalog_data, __end_idx, id_string):
    # check all matching keystring
    time_data = datalog_data['time'][:__end_idx]
    data_min, data_max = 0., 0.

    key_data = datalog_data[id_string][:__end_idx]
    ax.plot(time_data, key_data)
    # update min max for plotting
    data_min = min( data_min, min(i for i in key_data if i is not None) ) 
    data_max = max( data_max, max(i for i in key_data if i is not None) ) 
    # adjust time window
    ax.grid(True)
    ax.set(xlim= (time_data[0]-0.1, time_data[-1]+0.1), 
        ylim= (data_min-0.1, data_max+0.1))

class PredictGPAnimation():
    # WARNING: this class is not general 
    # and work only in the specific data within this script
    def __init__(self, datalog_data, __end_idx, robot_id, fname_output):
        self.__log = datalog_data
        self.__i = robot_id
        self.__cur_idx = 0

        from control_lib.GP_h import GP
        self.gp = GP( 
            SceneSetup.hypers_gp,
            SceneSetup.min_d_sample,
            SceneSetup.grid_size_plot,
            SceneSetup.dh_dt)

        # 2D PLOT parameter
        # ------------------------------------------------------------------------------------
        from matplotlib.gridspec import GridSpec
        rowNum, colNum = 2, 2 
        self.fig = plt.figure(figsize=(5*colNum, 3*rowNum), dpi= 100)
        gs = GridSpec( rowNum, colNum, figure=self.fig)

        self.__ax_gp = self.fig.add_subplot(gs[0:2,0:2])
        self.__ax_gp.set(xlabel="x [m]", ylabel="y [m]")
        self.__ax_gp.set_aspect('equal', adjustable='box', anchor='C')        

        # Display simulation time
        self.__drawn_time = self.__ax_gp.text(0.78, 0.99, 't = 0 s', color = 'k', fontsize='large', 
            horizontalalignment='left', verticalalignment='top', transform = self.__ax_gp.transAxes)

        __colorList = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.__robot_color = __colorList[self.__i]
        self.__gp_pl_pos, = self.__ax_gp.plot(0, 0, 'x', color=self.__robot_color)

        from nebolab_experiment_setup import NebolabSetup
        self.__field_x = NebolabSetup.FIELD_X
        self.__field_y = NebolabSetup.FIELD_Y        

        plt.tight_layout()

        # Execute the animation part
        import matplotlib.animation as animation
        # Step through simulation
        ani = animation.FuncAnimation(self.fig, self.loop_sequence, save_count=\
                                    __end_idx, interval = SimSetup.Ts*1000)
        print('saving animation for GP rob_'+str(self.__i))
        ani.save(fname_output, fps=round(1/SimSetup.Ts))
        print('Done. saved into '+fname_output)

    # main_loop to update the animation
    def loop_sequence(self, i = 0):
        # Extract data
        cur_data_X = self.__log['data_X_'+str(self.__i)][self.__cur_idx]
        cur_data_Y = self.__log['data_Y_'+str(self.__i)][self.__cur_idx]
        cur_data_N = self.__log['data_N_'+str(self.__i)][self.__cur_idx]
        cur_data_k = self.__log['data_k_'+str(self.__i)][self.__cur_idx]
        cur_posc_x = self.__log['posc_x_'+str(self.__i)][self.__cur_idx]
        cur_posc_y = self.__log['posc_y_'+str(self.__i)][self.__cur_idx]
        cur_posc = np.array([cur_posc_x, cur_posc_y, 0])
        cur_time = self.__log['time'][self.__cur_idx]

        # UPDATE 2D Plotting:
        self.__drawn_time.set_text('t = '+f"{cur_time:.1f}"+' s')
        self.__gp_pl_pos.set_data([cur_posc_x], [cur_posc_y])

        # if cur_data_N > 0:
        self.gp.data_X = cur_data_X
        self.gp.data_Y = cur_data_Y
        self.gp.N = cur_data_N
        self.gp.k = cur_data_k
        # update GP plot
        self.gp.draw_gp_whole_map_prediction( 
            self.__ax_gp, self.__field_x, self.__field_y, self.__i, cur_posc, SceneSetup.sense_dist, color=self.__robot_color)

        print('.',end='', flush=True)

        # increment index
        self.__cur_idx += 1


def scenario_pkl_plot():
    # ---------------------------------------------------
    with open(SimSetup.sim_fdata_log, 'rb') as f: visData = pickle.load(f)
    __stored_data = visData['stored_data']
    __end_idx = visData['last_idx']

    # set __end_idx manually
    # t_stop = 62
    # __end_idx = t_stop * ExpSetup.ROS_RATE

    # print(__stored_data['time'])
    # Print all key datas
    print('The file ' + SimSetup.sim_fdata_log + ' contains the following logs for ' + '{:.2f}'.format(__stored_data['time'][__end_idx]) + ' s:') 
    print(__stored_data.keys())

    # ---------------------------------------------------
    figure_short = (6.4, 3.4)
    figure_size = (6.4, 4.8)
    FS = 16 # font size
    LW = 1.5 # line width
    leg_size = 8

    #for robot_id in [0]: # range(SceneSetup.robot_num):
    for robot_id in range(SceneSetup.robot_num):

        # PLOT THE GP-CBF and minimum distance LIDAR
        # ---------------------------------------------------¨
        fig, ax = plt.subplots(1, figsize=figure_short)
        # plt.rcParams.update({'font.size': FS})
        plt.rcParams['text.usetex'] = True
        # plot
        #plot_pickle_individual_id(ax[0], __stored_data, __end_idx, 'h_gp_'+str(robot_id))
        plot_pickle_individual_id(ax, __stored_data, __end_idx, 'min_lidar_'+str(robot_id))
        ax.plot(__stored_data['time'][:__end_idx], [0.09 for _ in range(__end_idx)], linestyle='dashed', color='r', label='robot\'s radius')
        # label
        #ax[0].set_ylabel(r'$h$', fontsize=14)
        #ax[0].set_xlabel(r'$t$ [s]', fontsize=14)
        ax.set(ylim= (-0.1, SceneSetup.sense_dist+0.1))
        ax.set_xlabel(r'$t$ [s]', fontsize=14)
        ax.set_ylabel('min LIDAR [m]', fontsize=12)
        #ax[0].legend(loc= 'best', prop={'size': leg_size})
        ax.legend(loc= 'best', prop={'size': leg_size})
        #plt.show()
        figname = SimSetup.sim_defname+str(robot_id)+'_lidar_gp.pdf'
        pngname = SimSetup.sim_defname+str(robot_id)+'_lidar_gp.png'
        plt.savefig(figname, bbox_inches="tight", dpi=300)
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print( 'export figure: ' + figname, flush=True)

        plt.close('all')
        # PLOT THE GP-CBF and minimum distance LIDAR
        # ---------------------------------------------------¨
        fig, ax = plt.subplots(2, figsize=figure_short)
        # plt.rcParams.update({'font.size': FS})
        plt.rcParams['text.usetex'] = True
        # plot
        plot_pickle_individual_id(ax[0], __stored_data, __end_idx, 'h_gp_'+str(robot_id))
        plot_pickle_individual_id(ax[1], __stored_data, __end_idx, 'min_lidar_'+str(robot_id))
        ax[1].plot(__stored_data['time'][:__end_idx], [0.09 for _ in range(__end_idx)], linestyle='dashed', color='r', label='robot\'s radius')
        # label
        ax[0].set_ylabel(r'$h$', fontsize=14)
        #ax[0].set_xlabel(r'$t$ [s]', fontsize=14)
        ax[1].set(ylim= (-0.1, SceneSetup.sense_dist+0.6))
        ax[1].set_xlabel(r'$t$ [s]', fontsize=14)
        ax[1].set_ylabel('min LIDAR [m]', fontsize=14)
        ax[0].legend(loc= 'best', prop={'size': leg_size})
        ax[1].legend(loc= 'best', prop={'size': leg_size})
        #plt.show()
        figname = SimSetup.sim_defname+str(robot_id)+'_lidar_gp.pdf'
        pngname = SimSetup.sim_defname+str(robot_id)+'_lidar_gp.png'
        plt.savefig(figname, bbox_inches="tight", dpi=300)
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print( 'export figure: ' + figname, flush=True)

        plt.close('all')

        # PLOT THE U rectified in X and Y direction
        # ---------------------------------------------------¨
        fig, ax = plt.subplots(2, figsize=figure_short)
        # plt.rcParams.update({'font.size': FS})
        plt.rcParams['text.usetex'] = True
        # plot
        time_data = __stored_data['time'][:__end_idx]
        u_x_data = __stored_data['u_x_'+str(robot_id)][:__end_idx]
        u_y_data = __stored_data['u_y_'+str(robot_id)][:__end_idx]
        u_norm_data = __stored_data['u_norm_'+str(robot_id)][:__end_idx]

        ax[0].plot(time_data, u_x_data, color='b', label='$u_x$')
        ax[0].plot(time_data, u_y_data, color='r', label='$u_y$')
        ax[1].plot(time_data, u_norm_data, color='b', label='$\|u\|$')
        ax[1].plot(__stored_data['time'][:__end_idx], [0.1 for _ in range(__end_idx)], linestyle='dashed', color='r', label='$u_m/\sqrt 2$')
        # adjust grid and time window
        ax[0].grid(True)
        ax[0].set(xlim= (time_data[0]-0.1, time_data[-1]+0.1))
        ax[1].grid(True)
        ax[1].set(xlim= (time_data[0]-0.1, time_data[-1]+0.1))
        # label
        # ax[0].set( xlabel=r'$t$ [s]', ylabel=r'$u_x$[m/s]' )
        # ax[1].set( xlabel=r'$t$ [s]', ylabel=r'$u_y$[m/s]' )
        ax[0].set(ylabel=r'$u$[m/s]' )
        ax[1].set(xlabel=r'$t$ [s]', ylabel=r'$\|u\|$')
        # ax[0].set(ylim= (-0.1, SceneSetup.sense_dist+0.6))
        ax[0].set(ylim= (-1.1*SceneSetup.speed_limit, 1.1*SceneSetup.speed_limit))
        ax[1].set(ylim= (-0.01, 1.1*SceneSetup.speed_limit))
        
        ax[0].legend(loc= 'best', prop={'size': leg_size})
        ax[1].legend(loc= 'best', prop={'size': leg_size})
        #plt.show()
        figname = SimSetup.sim_defname+str(robot_id)+'_u_rect.pdf'
        pngname = SimSetup.sim_defname+str(robot_id)+'_u_rect.png'
        plt.savefig(figname, bbox_inches="tight", dpi=300)
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print( 'export figure: ' + figname, flush=True)

        plt.close('all')

        # Save the GP prediction animation
        # ---------------------------------------------------¨
        gp_fname_output = r''+SimSetup.sim_defname+str(robot_id)+'_gp_map.gif'
        PredictGPAnimation(__stored_data, __end_idx, robot_id, gp_fname_output)




def exp_video_pkl_plot():
    # def_folder = 'experiment_result/sim2D_obstacle_GP_v3/'
    def_folder = '../'
    videoloc = def_folder + 'ZED_record_20250614_202725.avi'
    outname = def_folder + 'snap_'

    # NOTE: the time is based on the video time. Which at the moment is a bit unsync. 
    # time_snap = [10, 60, 140, 240, 330, 413]  # in seconds
    time_snap = [0, 11, 19, 24]  # in seconds

    frame_shift = 0  # accomodate on-sync video and data, video ALWAYS earlier
    data_shift = 0 #int(5.8*ExpSetup.ROS_RATE) # accomodate delayed data (slow computation)

    robot_color = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']

    # Generate BGR color
    bgr_color = {}
    for i in range(SceneSetup.robot_num):
        h = robot_color[i].lstrip('#')
        r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        bgr_color[i] = (b, g, r)


    if videoloc is not None:
        import cv2
        from nebolab_experiment_setup import NebolabSetup        
        # -------------------------------
        # VIDEO DATA
        # -------------------------------
        # Initialize VIDEO
        cam = cv2.VideoCapture(videoloc)
        frame_per_second = cam.get(cv2.CAP_PROP_FPS)
        frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`


        # -------------------------------
        # RECORDED PICKLE DATA - for rectifying position or plot goals
        # -------------------------------
        # Initialize Pickle
        with open(SimSetup.sim_fdata_log, 'rb') as f:
            visData = pickle.load(f)
            print(visData['stored_data'].keys())
        __stored_data = visData['stored_data']
        __end_idx = visData['last_idx']

        goal_pxl = {i:np.zeros(2) for i in range(SceneSetup.robot_num) }

        print('Frames:', frame_count, ", Time:", visData['last_idx'])
        # print('Keys', __stored_data.keys())
        # SceneSetup.robot_color = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']

        # -------------------------------
        # IMAGE-BASED LOCALIZATION - for plotting trajectory
        # -------------------------------
        # Template for position 
        pos_pxl = {i:np.zeros((frame_count, 2)) for i in range(SceneSetup.robot_num) }
        # Initialized localization
        from scenarios.camerabased_localization import localize_from_ceiling
        localizer = localize_from_ceiling()

        # -------------------------------
        # VIDEO OUTPUT
        # -------------------------------
        out = cv2.VideoWriter(SimSetup.sim_defname + '_fixed.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_per_second, (width, height))
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions

        # -------------------------------
        # MAIN-LOOP OVER THE VIDEO FRAME
        # -------------------------------
        current_step = -frame_shift # syncing with the start of pickle data

        snap_iter = 0
        snap_timing_step = time_snap[snap_iter]*frame_per_second
        past_snap_step = 0 

        while True:
            # READ each frame
            ret, frame = cam.read()

            # ROLL frame until current_step = 0
            if current_step < 0:
                current_step += 1
                continue

            if ret:
                # Save position data from frame
                poses_center, poses_ahead = localizer.localize_all_robots(frame)
                frame = localizer.draw_pose(frame)

                for i in range(SceneSetup.robot_num):
                    is_valid_data = False
                    if poses_ahead[i] is not None:
                        pos_pxl[i][current_step,0], pos_pxl[i][current_step,1] = \
                            NebolabSetup.pos_m2pxl( poses_ahead[i].x, poses_ahead[i].y)
                        
                        # Error checking
                        if pos_pxl[i][current_step,0].is_integer(): 
                            # valid integer value, check distance from past data
                            dx = pos_pxl[i][current_step,0] - pos_pxl[i][current_step-1,0]
                            dy = pos_pxl[i][current_step,1] - pos_pxl[i][current_step-1,1]
                            dist = np.sqrt(dx**2 + dy**2)

                            # 0.01 m is the assumed max distance for 1 iteration in 30fps
                            is_valid_data = dist < 0.02 * NebolabSetup.SCALE_M2PXL
                        
                    # Invalid data, alternatively use last data
                    if current_step > 0 and not is_valid_data:
                        pos_pxl[i][current_step,0] = pos_pxl[i][current_step-1,0]
                        pos_pxl[i][current_step,1] = pos_pxl[i][current_step-1,1]

                # Get goal data from pickle
                time = current_step / frame_per_second
                idx = int(time * ExpSetup.ROS_RATE) + data_shift
                
                # TODO: plot goals, once it is known from last frame
                # for i in range(SceneSetup.robot_num):
                #     gx = __stored_data[f"goal_x_{i}"][idx]
                #     gy = __stored_data[f"goal_y_{i}"][idx]
                #     goal_pxl[i][0], goal_pxl[i][1] = NebolabSetup.skewed_pos_m2pxl(gx, gy)

                
                # save if snap
                if current_step > snap_timing_step:
                    
                    frame_snap = frame.copy()
                    snap_name = outname + str(time_snap[snap_iter]) + '.jpg'

                    # generate_snap 
                    line_width = 8
                    for i in range(SceneSetup.robot_num):
                        for step in range(past_snap_step+1, current_step):
                            pxl_from = ( int(pos_pxl[i][step-1,0]), int(pos_pxl[i][step-1,1]) )
                            pxl_to = ( int(pos_pxl[i][step,0]), int(pos_pxl[i][step,1]) )
                            # draw line segment
                            cv2.line(frame_snap, pxl_from, pxl_to, bgr_color[i], line_width)

                    cv2.imwrite(snap_name, frame_snap)
                    print('exporting snap: ' + snap_name, flush=True)

                    # advance snap timing
                    past_snap_step = current_step
                    snap_iter += 1 
                    if snap_iter < len(time_snap):
                        snap_timing_step = time_snap[snap_iter]*frame_per_second
                    else:
                        snap_timing_step = frame_count + 1

                # save video
                cv2.imshow('frame', frame)
                key = cv2.waitKey(1) & 0xFF == ord('q')
                out.write(frame)

                # Advance step
                current_step += 1

            else:
                break
        
        out.release()
        cam.release()
        cv2.destroyAllWindows()


