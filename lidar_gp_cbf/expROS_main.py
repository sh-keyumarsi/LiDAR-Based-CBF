import rospy, signal
from geometry_msgs.msg import Pose2D, Twist
from sensor_msgs.msg import LaserScan

#from scenarios_unicycle.CCTA2022_Controller import FeedbackInformation, Controller, ControlOutput, SceneSetup
#from scenarios_unicycle.CCTA2022_FormationObstacle_scenario import ExpSetup, ExperimentEnv

# from scenarios_unicycle.sim2D_basic_scene import FeedbackInformation, Controller, ControlOutput, SceneSetup, ExpSetup, ExperimentEnv
# from scenarios_unicycle.sim2D_cbf_lidar_simple import FeedbackInformation, Controller, ControlOutput, SceneSetup, ExpSetup, ExperimentEnv

# from scenarios_unicycle.Resilient_scenario import FeedbackInformation, Controller, ControlOutput, SceneSetup, ExpSetup, ExperimentEnv


from scenarios.sim2D_obstacle_GP import FeedbackInformation, Controller, ControlOutput, SceneSetup, ExpSetup, ExperimentEnv


# TODO: think of a better way to do this with less than 4 robot

class Computation():
    def __init__(self):
        # Initialize components
        self.environment = ExperimentEnv() # Always the first to call, define main setup
        self.controller_block = Controller() 
        # Initialize messages to pass between blocks
        self.feedback_information = FeedbackInformation() 
        self.control_input = ControlOutput() 

        # INITIALIZE ROS SUBSCRIBER and Publisher
        robot_names =['tb3_0','tb3_1','tb3_2','tb3_3'] # TODO: this is restrictive at the moment
        rospy.Subscriber('/{}/posc'.format(robot_names[0]), Pose2D, self.environment.posc_callback_0) # red
        rospy.Subscriber('/{}/posc'.format(robot_names[1]), Pose2D, self.environment.posc_callback_1) # blue
        rospy.Subscriber('/{}/posc'.format(robot_names[2]), Pose2D, self.environment.posc_callback_2) # green
        rospy.Subscriber('/{}/posc'.format(robot_names[3]), Pose2D, self.environment.posc_callback_3) # orange
        # another one for look ahead pos
        rospy.Subscriber('/{}/pos'.format(robot_names[0]), Pose2D, self.environment.pos_callback_0)
        rospy.Subscriber('/{}/pos'.format(robot_names[1]), Pose2D, self.environment.pos_callback_1)
        rospy.Subscriber('/{}/pos'.format(robot_names[2]), Pose2D, self.environment.pos_callback_2)
        rospy.Subscriber('/{}/pos'.format(robot_names[3]), Pose2D, self.environment.pos_callback_3)
        # LIDAR scan data
        rospy.Subscriber('/{}/scan'.format(robot_names[0]), LaserScan, self.environment.scan_LIDAR_callback_0)
        rospy.Subscriber('/{}/scan'.format(robot_names[1]), LaserScan, self.environment.scan_LIDAR_callback_1)
        rospy.Subscriber('/{}/scan'.format(robot_names[2]), LaserScan, self.environment.scan_LIDAR_callback_2)
        rospy.Subscriber('/{}/scan'.format(robot_names[3]), LaserScan, self.environment.scan_LIDAR_callback_3)

        self.ros_pubs = []
        for i in range(SceneSetup.robot_num): 
            self.ros_pubs += [rospy.Publisher('/{}/cmd_vel'.format(robot_names[i]),Twist, queue_size=1)]

        # Add handler if CTRL+C is pressed --> then save data to pickle
        signal.signal(signal.SIGINT, self.signal_handler)

    # MAIN LOOP CONTROLLER & VISUALIZATION
    def loop_sequence(self, it = 0):
        # Showing Time Stamp
        if (it > 0) and (it % ExpSetup.ROS_RATE == 0):
            t = round(it/ExpSetup.ROS_RATE)
            print('Experiment t = {}s.'.format(t))

        # Checking all is not None is restrictive for now, does not work if we only use 1 robot
        # TODO: test this, it is more flexible now, but we need to use the robot in the correct order
        # according to its numbering, otherwise change the name ordering
        if all (v is not None for v in self.environment.global_poses[ :SceneSetup.robot_num ]):
            # Condense Feedback, compute control input, and propagate control
            self.environment.update_feedback( self.feedback_information )
            self.controller_block.compute_control( self.feedback_information, self.control_input )
        
            for i in range(SceneSetup.robot_num): 
                v_lin, omega = self.environment.get_i_vlin_omega( i, self.control_input )
                self.ros_pubs[i].publish( self.si_to_TBTwist(v_lin, omega) )

            self.environment.update_log( self.control_input )
            

    @staticmethod
    def si_to_TBTwist(vel_lin, vel_ang):  
        # TODO: do max (or saturation) if needed
        TBvel = Twist()
        TBvel.linear.x = vel_lin
        TBvel.linear.y = 0
        TBvel.linear.z = 0
        TBvel.angular.x = 0
        TBvel.angular.y = 0
        TBvel.angular.z = vel_ang
        return TBvel

    # Allow CTRL+C to stop the controller and dump the log into pickle
    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C. Turning off the controller.')

        self.environment.save_log_data()

        # LOG the data from simulation
        # if save_data:
        #     print('Storing the data to files...', flush=True)
        #     with open(fdata_vis, 'wb') as f:
        #         pickle.dump(dict(data_log=self.data_log), f)
        #     print('Done.')
        exit() # Force Exit


def main():
    comp = Computation()
    try:
        rospy.init_node('controller', anonymous=True)
        r = rospy.Rate(ExpSetup.ROS_RATE)
        
        it = 0
        while not rospy.is_shutdown():

            comp.loop_sequence(it)
            it +=1
            r.sleep()

    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()