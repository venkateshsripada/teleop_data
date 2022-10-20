#!/usr/bin/env python3

import rospy
import cv2
import sys
import numpy as np
import pickle as pl
import message_filters
import tf.transformations
from PIL import Image as pil_im

import moveit_commander
import moveit_msgs.msg

from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from franka_msgs.msg import FrankaState
from franka_msgs.msg import ErrorRecoveryAction, ErrorRecoveryActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Brings in the SimpleActionClient
import actionlib

import dynamic_reconfigure.client

from controller_manager_msgs.srv import SwitchController

class DataCollection:
    def __init__(self):
        rospy.init_node("data_collector_node")
        self.bridge = CvBridge()

        self.init_top_cam = False
        self.init_arm_cam = False 
        self.at_intital_position = False
        self.controller_switched = False
        self.follower_joint_states_arr = False
        self.follower_task_states_arr = False
        self.stop = False

        rospy.loginfo("Your wish is my command")

        self.leaderComplinaceConfigServer = dynamic_reconfigure.client.Client("/leader/leader_cartesian_impedance_controller_controller_params", timeout=30, config_callback=self.compliance_config_cb)
        self.franka_subscribers()
        self.franka_publishers()
        self.main()

    def franka_subscribers(self):
        top_cam_sub = message_filters.Subscriber("/top_camera/color/image_raw", Image)
        arm_cam_sub = message_filters.Subscriber("/arm_camera/color/image_raw", Image)
        side_cam_sub = message_filters.Subscriber("/side_camera/color/image_raw", Image)
        joint_state_sub = message_filters.Subscriber("/panda_follower/panda_follower_state_controller/joint_states", JointState)
        task_space_sub = message_filters.Subscriber("/panda_follower/panda_follower_state_controller/franka_states", FrankaState)

        ts = message_filters.ApproximateTimeSynchronizer([top_cam_sub, arm_cam_sub, side_cam_sub, joint_state_sub, task_space_sub], queue_size=10, slop=0.05)
        ts.registerCallback(self.callback)

    def franka_publishers(self):
        self.follower_arm_recovery_client = actionlib.SimpleActionClient( "/panda_follower/franka_control/error_recovery", ErrorRecoveryAction )
        self.leader_joint_controller_pub = rospy.Publisher( "/leader/position_joint_trajectory_controller/command", JointTrajectory, queue_size = 1 )

    def callback(self, top_cam, arm_cam, side_cam, joint_sub, task_sub):
        self.follower_joint_states.append(np.asarray(joint_sub.position))
        self.follower_task_states.append(task_sub.O_T_EE)

        try:
            cv_image = self.bridge.imgmsg_to_cv2(arm_cam, "bgr8")
            cv_image_top = self.bridge.imgmsg_to_cv2(top_cam, "bgr8")
            cv_image_side = self.bridge.imgmsg_to_cv2(side_cam, "bgr8")
        except CvBridgeError as e:
            print("CvBridgeError: ", e)

        self.arm_cam.append(np.asarray(cv_image))
        self.top_cam.append(np.asarray(cv_image_top))
        self.side_cam.append(np.asarray(cv_image_side))

    def recovery_follower_arm_error(self): 
        # This function will recover the franka robot in case it is stuck

        goal = ErrorRecoveryActionGoal() 
        self.follower_arm_recovery_client.send_goal(goal)

    def save_to_file(self, file, array):
        np.save(file, array)

    def initial_position_trajectory(self):
        traj_point = JointTrajectoryPoint()
        trajectory = JointTrajectory()

        leader_joint_names = []

        for i in range(7):
            leader_joint_name = "leader_joint" + str(i+1)
            leader_joint_names.append( leader_joint_name ) 

        leader_joint_values = [-0.12, -0.45, -0.06, -2.39, -0.02, 1.96, 0.85]

        traj_point.positions = leader_joint_values
        traj_point.time_from_start = rospy.Duration( 1.0 )
        trajectory.header.stamp = rospy.Time().now()
        trajectory.joint_names = leader_joint_names
        trajectory.points.append( traj_point )

        self.leader_joint_controller_pub.publish( trajectory )

        self.at_intital_position = True

    def compliance_config_cb(self, config):
        self.leaderComplianceParams = config

    def switch_controllers(self):
        try:
            switch_controller = rospy.ServiceProxy('/leader/controller_manager/switch_controller', SwitchController)
            ret = switch_controller(start_controllers=['leader_cartesian_impedance_controller'], stop_controllers=['position_joint_trajectory_controller'])
            print("=================YOU CAN CONTROL===============")
            self.controller_switched = True
        except rospy.ServiceException as e:
            print ("Service call failed: %s",e)

    def read_keystrokes(self):
        image = np.zeros(  ( 200, 200 ) , dtype = np.uint8)
        cv2.imshow( "Switch Control", image ) 
        k = cv2.waitKey(30) 
        if k == 27:
            self.stop = True
            cv2.destroyAllWindows()
        elif k < 0:
            pass


    def post_processing(self):
        self.save_to_file("arm_camera_" + sys.argv[1] + ".npy", self.arm_cam)
        self.save_to_file("top_camera_" + sys.argv[1] + ".npy", self.top_cam)
        self.save_to_file("side_camera_" + sys.argv[1] + ".npy", self.side_cam)
        self.save_to_file("joint_states_" + sys.argv[1] + ".npy", self.follower_joint_states)

        follower_task_states_mod = []
        for i in range(len(self.follower_task_states)):
            current_quaternion = tf.transformations.quaternion_from_matrix(np.transpose(np.reshape(self.follower_task_states[i], (4, 4))))
            current_quaternion = current_quaternion / np.linalg.norm(current_quaternion)
            orientation_x = current_quaternion[0]
            orientation_y = current_quaternion[1]
            orientation_z = current_quaternion[2]
            orientation_w = current_quaternion[3]
            position_x = self.follower_task_states[i][12]
            position_y = self.follower_task_states[i][13]
            position_z = self.follower_task_states[i][14]
            follower_task_states_mod.append(np.array([orientation_x, orientation_y, orientation_z, orientation_w, position_x, position_y, position_z]))
        self.save_to_file("task_space_" + sys.argv[1] + ".npy", follower_task_states_mod)

    def main(self):
        # rate = rospy.Rate(30)   # 30hz
        start = rospy.get_time()
        while not rospy.is_shutdown():
            if self.at_intital_position == False:
                self.initial_position_trajectory()
            elif (self.at_intital_position==True) and (self.controller_switched== False):
                self.switch_controllers()

            if (self.init_top_cam == False) or (self.init_arm_cam == False) or\
               (self.follower_joint_states_arr == False) or (self.follower_task_states_arr == False):
                
                self.top_cam = []
                self.arm_cam = []
                self.side_cam = []
                self.follower_joint_states = []
                self.follower_task_states = []
                
                self.frames = []
                self.init_top_cam = True
                self.init_arm_cam = True
                self.follower_joint_states_arr = True
                self.follower_task_states_arr = True
            else:
                try:
                    self.recovery_follower_arm_error()
                    self.read_keystrokes()
                    end = rospy.get_time()
                    """
                    The following if statement stops the code and records meta data
                    The meta data is a np array with three values
                    # 0 - The total time taken to perform the task
                    # 1 - The total number of frames captured
                    # 2 - The frame rate i.e number of frames captured in a second
                    """
                    if self.stop:
                        exp_time = end - start
                        self.frames.append(exp_time)
                        self.frames.append(len(self.arm_cam))
                        frame_rate = len(self.arm_cam) / exp_time
                        self.frames.append(frame_rate)
                        self.save_to_file("meta_" + sys.argv[1] + ".npy", self.frames)
                        self.post_processing()
                        break
                except KeyboardInterrupt:
                    break
            # rate.sleep()


data = DataCollection()