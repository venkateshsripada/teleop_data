#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import tf.transformations
from PIL import Image as pil_im

from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from franka_msgs.msg import FrankaState
from franka_msgs.msg import ErrorRecoveryAction, ErrorRecoveryActionGoal

# Brings in the SimpleActionClient
import actionlib

class DataCollection:
    def __init__(self):
        rospy.init_node("data_collector_node")
        self.bridge = CvBridge()

        self.init_top_cam = False
        self.init_arm_cam = False 
        self.follower_joint_states_arr = False
        self.follower_task_states_arr = False

        rospy.loginfo("Your wish is my command")
        self.franka_subscribers()
        self.franka_publishers()
        self.main()

    def franka_subscribers(self):
        rospy.Subscriber("/top_camera/color/image_raw", Image, self.top_camera_cb)
        rospy.Subscriber("/arm_camera/color/image_raw", Image, self.arm_camera_cb)
        rospy.Subscriber("/panda_follower/panda_follower_state_controller/joint_states", JointState, self.joint_states_cb)
        rospy.Subscriber("/panda_follower/panda_follower_state_controller/franka_states", FrankaState, self.franka_states_cb)

    def franka_publishers(self):
        self.follower_arm_recovery_client = actionlib.SimpleActionClient( "/panda_follower/franka_control/error_recovery", ErrorRecoveryAction )

    def recovery_follower_arm_error(self): 
        # This function will recover the franka robot in case it is stuck

        goal = ErrorRecoveryActionGoal() 
        self.follower_arm_recovery_client.send_goal(goal)

    def joint_states_cb(self, msg):
        """
        This function saves data of the franka arm's joint values
        """
        self.follower_joint_states.append(np.asarray(msg.position))
        self.save_to_file("joint_states.npy", self.follower_joint_states)

    def franka_states_cb(self, msg):
        """
        This function saves data of the franka arm's task space position and orientation
        The array is of form [orientation_x, orienttation_y, orientation_z, orientation_w, pos_x, pos_y, pos_z]
        """
        current_quaternion = tf.transformations.quaternion_from_matrix(np.transpose(np.reshape(msg.O_T_EE, (4, 4))))
        current_quaternion = current_quaternion / np.linalg.norm(current_quaternion)

        orientation_x = current_quaternion[0]
        orientation_y = current_quaternion[1]
        orientation_z = current_quaternion[2]
        orientation_w = current_quaternion[3]
        position_x = msg.O_T_EE[12]
        position_y = msg.O_T_EE[13]
        position_z = msg.O_T_EE[14]
        self.follower_task_states.append([orientation_x, orientation_y, orientation_z, orientation_w, position_x, position_y, position_z])
        self.save_to_file("task_space.npy", self.follower_task_states)

    def arm_camera_cb(self, msg):
        """
        This function will save the data of the camera in hand as a numpy array
        The shape will be n x 480 x 640 x 3 where n is the number of samples taken by the camera
        """

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print("CvBridgeError: ", e)
        self.arm_cam.append(np.asarray(cv_image))
        self.save_to_file("arm_camera.npy", self.arm_cam)

    def top_camera_cb(self, msg):
        """
        This function will save the data of the top camera as a numpy array 
        The shape will be n x 480 x 640 x 3 where n is the number of samples taken by the camera
        """

        try:
            cv_image_top = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print("CvBridgeError: ", e)
        self.top_cam.append(np.asarray(cv_image_top))
        self.save_to_file("top_camera.npy", self.top_cam)

    def save_to_file(self, file, array):
        np.save(file, array)

    def load_file(self):
        joints = np.load('joint_states.npy')
        task = np.load('task_space.npy')
        tc = np.load('top_camera.npy')
        ac = np.load('arm_camera.npy')
        # for i in range(ac.shape[0]):
        #     img = pil_im.fromarray(ac[i], 'RGB')
        #     img.show()
        print("Top cam: ", tc.shape)
        print("Arm cam: ",ac.shape)
        print("Joint vals: ",joints.shape)
        print("Task vals: ", task.shape)

    def main(self):
        while not rospy.is_shutdown():
            if (self.init_top_cam == False) or (self.init_arm_cam == False) or\
               (self.follower_joint_states_arr == False) or (self.follower_task_states_arr == False):
                
                self.top_cam = []
                self.arm_cam = []
                self.follower_joint_states = []
                self.follower_task_states = []
                self.init_top_cam = True
                self.init_arm_cam = True
                self.follower_joint_states_arr = True
                self.follower_task_states_arr = True
            else:
                try:
                    self.recovery_follower_arm_error()
                except KeyboardInterrupt:
                    break


data = DataCollection()
data.load_file()