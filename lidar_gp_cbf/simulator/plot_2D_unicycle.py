import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class draw2DUnicyle():
    def __init__(self, ax, init_pos, init_theta = None, field_x = None, field_y = None, pos_trail_nums = None):
        # The state should be n rows and 3 dimensional column (pos.X, pos.Y, and theta)
        # pos_trail_nums determine the number of past data to plot as trajectory trails
        self.__ax = ax
        self.__ax.set(xlabel="x [m]", ylabel="y [m]")
        self.__ax.set_aspect('equal', adjustable='box', anchor='C')
        # Set field
        if field_x is not None: self.__ax.set(xlim=(field_x[0]-0.1, field_x[1]+0.1))
        if field_y is not None: self.__ax.set(ylim=(field_y[0]-0.1, field_y[1]+0.1))

        self.__robot_num = init_pos.shape[0] # row number
        if init_theta is None: init_theta = np.zeros(self.__robot_num)
        
        # Plotting variables
        self.__patch_info = {i:None for i in range(self.__robot_num)}
        self.__colorList = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.__icon_id = 2 # unicycle # TODO: remove the omnidirectional?

        # Prepare buffer for the trail
        self.__pl_trail = [None]*self.__robot_num
        if pos_trail_nums is not None:        
            self.__trail_data = {i:None for i in range(self.__robot_num)}
            
            for i in range(self.__robot_num):
                if pos_trail_nums[i] > 0:            
                    # use initial position to populate all matrix (pos_trail_nums-row x dim-col)
                    # theta is not used for plotting the trail
                    self.__trail_data[i] = np.tile(init_pos[i], (pos_trail_nums[i], 1))
    
                    # Plot the first data
                    self.__pl_trail[i], = self.__ax.plot(
                        self.__trail_data[i][:,0], self.__trail_data[i][:,1], 
                    '--', color=self.__colorList[i])

        # Plot the data
        self.update(init_pos, init_theta)
    

    def update(self, all_pos, all_theta):
        for i in range(self.__robot_num):
            self.__draw_icon( i, np.array([all_pos[i,0], all_pos[i,1], all_theta[i]], dtype=object), arrow_col=self.__colorList[i])

            if self.__pl_trail[i] is not None: # update trail data
                # roll the data, fill the new one from the top and then update plot
                self.__trail_data[i] = np.roll(self.__trail_data[i], self.__trail_data[i].shape[1])
                self.__trail_data[i][0,:] = all_pos[i,:]
                self.__pl_trail[i].set_data(self.__trail_data[i][:,0], self.__trail_data[i][:,1])


    def __draw_icon(self, id, robot_state, arrow_col = 'b'): # draw mobile robot as an icon
        # Extract data for plotting
        px = robot_state[0]
        py = robot_state[1]
        th = robot_state[2]
        # Basic size parameter
        scale = 1
        body_rad = 0.08 * scale # m
        wheel_size = [0.1*scale, 0.02*scale] 
        arrow_size = body_rad
        # left and right wheels anchor position (bottom-left of rectangle)
        if self.__icon_id == 2: thWh = [th+0., th+np.pi] # unicycle
        else: thWh = [ (th + i*(2*np.pi/3) - np.pi/2) for i in range(3)] # default to omnidirectional
        thWh_deg = [np.rad2deg(i) for i in thWh]
        wh_x = [ px - body_rad*np.sin(i) - (wheel_size[0]/2)*np.cos(i) + (wheel_size[1]/2)*np.sin(i) for i in thWh ]
        wh_y = [ py + body_rad*np.cos(i) - (wheel_size[0]/2)*np.sin(i) - (wheel_size[1]/2)*np.cos(i) for i in thWh ]
        # Arrow orientation anchor position
        ar_st= [px, py] #[ px - (arrow_size/2)*np.cos(th), py - (arrow_size/2)*np.sin(th) ]
        ar_d = (arrow_size*np.cos(th), arrow_size*np.sin(th))
        # initialized unicycle icon at the center with theta = 0
        if self.__patch_info[id] is None: # first time drawing
            self.__patch_info[id] = [None]*(2+len(thWh))
            self.__patch_info[id][0] = self.__ax.add_patch( plt.Circle( (px, py), body_rad, color='#AAAAAAAA') )
            self.__patch_info[id][1] = self.__ax.quiver( ar_st[0], ar_st[1], ar_d[0], ar_d[1], 
                scale_units='xy', scale=1, color=arrow_col, width=0.1*arrow_size)
            for i in range( len(thWh) ):
                self.__patch_info[id][2+i] = self.__ax.add_patch( plt.Rectangle( (wh_x[i], wh_y[i]), 
                    wheel_size[0], wheel_size[1], angle=thWh_deg[i], color='k') )
        else: # update existing patch
            self.__patch_info[id][0].set( center=(px, py) )
            self.__patch_info[id][1].set_offsets( ar_st )
            self.__patch_info[id][1].set_UVC( ar_d[0], ar_d[1] )
            for i in range( len(thWh) ):
                self.__patch_info[id][2+i].set( xy=(wh_x[i], wh_y[i]) )
                self.__patch_info[id][2+i].angle = thWh_deg[i]

    # Additional procedure to extract each robot trajectory, so it can be reused in another plot
    def extract_robot_i_trajectory(self, robot_id): return self.__trail_data[robot_id]



class drawMovingEllipse():
    def __init__(self, ax, pos_xy, major_l, minor_l, theta=0., alpha=0.2, col = 'k'):
        self.__ellipse = Ellipse((pos_xy[0], pos_xy[1]), major_l, minor_l, angle=theta, alpha=alpha)
        ax.add_artist(self.__ellipse)

    def update(self, pos_xy, theta):
        self.__ellipse.center = (pos_xy[0], pos_xy[1])
        self.__ellipse.angle = np.rad2deg(theta)

