#!/usr/bin/python3
import rclpy, signal
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from geometry_msgs.msg import Pose2D, Twist
from sensor_msgs.msg import LaserScan

from functools import partial

from .scenarios.sim2D_obstacle_GP import FeedbackInformation, Controller, ControlOutput, SceneSetup
from .scenarios.sim2D_obstacle_GP import ExpSetup, ExperimentEnv


# from scenarios_unicycle.sim2D_basic_scene import FeedbackInformation, Controller, ControlOutput, SceneSetup, ExpSetup, ExperimentEnv
# from .scenarios_unicycle.sim2D_cbf_lidar_simple import FeedbackInformation, Controller, ControlOutput, SceneSetup, ExpSetup, ExperimentEnv

# from scenarios_unicycle.Resilient_scenario import FeedbackInformation, Controller, ControlOutput, SceneSetup, ExpSetup, ExperimentEnv

# TODO: think of a better way to do this with less than 4 robot

# # SERVICE:
# from std_srvs.srv import SetBool

# ros2 service call /set_recording std_srvs/srv/SetBool "data: True"
# ros2 service call /set_recording std_srvs/srv/SetBool "data: False"

class Computation(Node):
    def __init__(self, ROS_NODE_NAME):
        super().__init__(ROS_NODE_NAME)

        # Initialize components
        self.environment = ExperimentEnv()  # Always the first to call, define main setup
        self.controller_block = Controller()
        # Initialize messages to pass between blocks
        self.feedback_information = FeedbackInformation()
        self.control_input = ControlOutput()

        # ROS
        self.ros_pubs = {}
        self.stop = False
        self.sensor_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        for robot_index in range(SceneSetup.robot_num):
            #robot_names =['tb3_0','tb3_1','tb3_2','tb3_3']

            tb_name = f'tb3_{robot_index}'

            # Create pose ahead subscribers
            self.get_logger().info(f'Creating pos ahead subscriber /{tb_name}/pos')
            self.pos_sub = self.create_subscription(Pose2D,
                                                    f'/{tb_name}/pos',
                                                    partial(self.environment.pos_callback, index=robot_index),
                                                    qos_profile=qos_profile_sensor_data,
                                                    callback_group=self.sensor_cb_group)
            self.pos_sub

            # Create pose center subscribers
            self.get_logger().info(f'Creating pos center subscriber /{tb_name}/pos')
            self.posc_sub = self.create_subscription(Pose2D,
                                                    f'/{tb_name}/posc',
                                                    partial(self.environment.posc_callback, index=robot_index),
                                                    qos_profile=qos_profile_sensor_data,
                                                    callback_group=self.sensor_cb_group)
            self.posc_sub

            # # Create LiDAR subscribers
            # self.get_logger().info(f'Creating LiDAR data subscriber: /{tb_name}/scan')
            # self.create_subscription(LaserScan,
            #                          f'/{tb_name}/scan',
            #                          partial(self.environment.scan_LIDAR_callback, index=robot_index),
            #                          qos_profile=qos_profile_sensor_data,
            #                          callback_group=self.sensor_cb_group)

            # Create LiDAR filtered subscribers
            self.get_logger().info(f'Creating LiDAR data subscriber: /{tb_name}/scan_filtered')
            self.lidar_sub = self.create_subscription(LaserScan,
                        f'/{tb_name}/scan_filtered',
                        partial(self.environment.scan_LIDAR_callback, index=robot_index),
                        qos_profile=qos_profile_sensor_data,
                        callback_group=self.sensor_cb_group)
            self.lidar_sub


            # create cmd_vel publisher
            self.get_logger().info(f'Creating cmd_vel publisher: /{tb_name}/cmd_vel')
            self.ros_pubs[tb_name] = self.create_publisher(Twist, '/{}/cmd_vel'.format(tb_name), 1)


        # # SERVICE FOR RECORDING:
        # self.cli = self.create_client(SetBool, 'set_recording')
        # self.req = SetBool.Request()
        # self.send_request(True)

        # Add handler if CTRL+C is pressed --> then save data to pickle
        signal.signal(signal.SIGINT, self.signal_handler)

        # Set timer for controller loop in each iteration
        self.Ts = 1. / ExpSetup.ROS_RATE
        self.controller_timer = self.create_timer(self.Ts, self.control_loop,
                                                  callback_group=self.timer_cb_group)
        self.it = 0
        self.check_t = self.time()

    # SERVICE FOR RECORDING:
    def send_request(self, state):
        self.get_logger().info(f"Sending request to set_recording with state: {state}")
        self.req.data = state
        self.future = self.cli.call_async(self.req)

    def time(self):
        """Returns the current time in seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    # MAIN LOOP CONTROLLER & VISUALIZATION
    def control_loop(self):

        now = self.time()
        diff = (now - self.check_t)
        if diff > (1.1 * self.Ts):  # Add 10% extra margin
            self.get_logger().info(
                'WARNING loop rate is slower than expected. Period (ms): {:0.2f}'.format(diff * 1000))
        self.check_t = now

        # Showing Time Stamp
        if (self.it > 0) and (self.it % ExpSetup.ROS_RATE == 0):
            t = self.it * self.Ts
            #print('Experiment t = {}s.'.format(t))
        self.it += 1

        # Checking all is not None is restrictive for now, does not work if we only use 1 robot
        # TODO: test this, it is more flexible now, but we need to use the robot in the correct order
        # according to its numbering, otherwise change the name ordering
        if all(v is not None for v in self.environment.global_poses[:SceneSetup.robot_num]):
            # Condense Feedback, compute control input, and propagate control
            self.environment.update_feedback(self.feedback_information)
            self.controller_block.compute_control(self.feedback_information, self.control_input)

            for i in range(SceneSetup.robot_num):
                if self.stop:
                    print(f"STOP ROBOT: {i}")
                    v_lin = 0.0
                    omega = 0.0
                else:
                    v_lin, omega = self.environment.get_i_vlin_omega(i, self.control_input)
                self.ros_pubs[f"tb3_{i}"].publish(self.si_to_TBTwist(v_lin, omega))

            self.environment.update_log(self.control_input)

    @staticmethod
    def si_to_TBTwist(vel_lin, vel_ang):
        # TODO: do max (or saturation) if needed
        TBvel = Twist()
        TBvel.linear.x = vel_lin
        TBvel.linear.y = 0.0
        TBvel.linear.z = 0.0
        TBvel.angular.x = 0.0
        TBvel.angular.y = 0.0
        TBvel.angular.z = vel_ang
        return TBvel

    # Allow CTRL+C to stop the controller and dump the log into pickle
    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C. Turning off the controller.')

        # LOG the data from simulation
        # if save_data:
        #     print('Storing the data to files...', flush=True)
        #     with open(fdata_vis, 'wb') as f:
        #         pickle.dump(dict(data_log=self.data_log), f)
        #     print('Done.')

        # # STOP RECORDING
        # self.send_request(False)

        # Stop all robots at the end
        for i in range(SceneSetup.robot_num):
            self.ros_pubs[f"tb3_{i}"].publish( self.si_to_TBTwist(0., 0.) )

        self.stop = True

        self.environment.save_log_data()

        exit()  # Force Exit


def main(args=None):
    rclpy.init(args=args)
    node = Computation(ExpSetup.ROS_NODE_NAME)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        node.get_logger().info("Spinning " + ExpSetup.ROS_NODE_NAME)
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
