#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np
import copy

from geometry_msgs.msg import PoseStamped, Twist
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState
from franka_msgs.msg import ErrorRecoveryAction, ErrorRecoveryActionGoal
import franka_gripper.msg
import control_msgs.msg 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import cv2

# Brings in the SimpleActionClient
import actionlib


import dynamic_reconfigure.client


class PandaToPandaTeleoperation:
    """ this class subscribe to joint states of leader and follower 
        and moves the follower arm related to leader arm.

        Optionally: 
            1) Provide haptics feedback 
            2) Provide virtual wall 
    """

    def __init__(self, leader_ns = "leader", follower_ns = "panda_follower" ): 

        rospy.init_node("panda_panda_teleoperation") 

        self.leader_ns = leader_ns
        self.follower_ns = follower_ns
        
        rospy.loginfo("starting")
        self._teleoperation_variables()
        self._teleoperation_subscribers() 
        self._teleoperation_publishers() 
        # rospy.spin()
        self.control_loop()

    def _teleoperation_variables(self): 
 
        self.leader_joint_state = None
        self.follower_joint_state = None

        self.follower_joint_controller_pub = None
        self.leader_compliance_controller = None
        self.leader_compliance_config = None
        
    def _teleoperation_subscribers(self): 

        rospy.Subscriber( self.leader_ns + "/joint_states" , JointState, self.leader_joint_state_cb )
        rospy.Subscriber( self.follower_ns + "/joint_states" , JointState, self.follower_joint_state_cb )

    def _teleoperation_publishers(self): 

        self.follower_joint_controller_pub = rospy.Publisher( "/panda_follower/panda_follower_joint_trajectory_controller/command", JointTrajectory, queue_size = 1 )
        self.follower_arm_recovery_client = actionlib.SimpleActionClient( "/panda_follower/franka_control/error_recovery", ErrorRecoveryAction )
        self.follower_arm_recovery_client.wait_for_server()

    def leader_joint_state_cb (self, msg ): 
        self.leader_joint_state = msg  
 
    def follower_joint_state_cb (self, msg ): 
        self.follower_joint_state = msg  
  
    def create_joint_trajectory(self): 

        traj_point = JointTrajectoryPoint()
        trajectory = JointTrajectory()
        
        follower_joint_names = []
        Kp = 0.5    #1.3
        Kd = 0.02   #0.03
        dt = 1.6    #1.75

        for i in range(7):
            follower_joint_name = self.follower_ns + "_joint" + str(i+1)
            follower_joint_names.append( follower_joint_name ) 

        # leader_joint_values = self.leader_task_state.q 
        # follower_joint_values = self.follower_task_state.q 
        leader_joint_values = self.leader_joint_state.position[:7]
        follower_joint_values = self.follower_joint_state.position[:7]


        position_error = [ i - j for i, j in zip(leader_joint_values, follower_joint_values ) ]
        velocity_error = self.leader_joint_state.velocity[:7]  #self.leader_task_state.dq
    
        correction = [ Kp * p_err + Kd * v_err for p_err, v_err in zip( position_error, velocity_error ) ]
        traj_point.positions = [ i + j for i, j in zip( follower_joint_values , correction ) ]
        traj_point.velocities = velocity_error

        traj_point.time_from_start = rospy.Duration( dt )
        trajectory.header.stamp = rospy.Time().now()
        trajectory.joint_names = follower_joint_names
        trajectory.points.append( traj_point )

        return trajectory

    def apply_leader_position_to_follower(self): 
        
        trajectory = self.create_joint_trajectory()
        self.follower_joint_controller_pub.publish( trajectory )
 
    def recovery_follower_arm_error(self): 
        goal = ErrorRecoveryActionGoal() 
        self.follower_arm_recovery_client.send_goal(goal)
        # self.follower_arm_recovery_client.wait_for_result() 


    def control_loop(self): 
        
        move_gripper_to_leader_gripper = False

        rospy.loginfo("Press Ctrl+C if anything goes sideways")
        rate = rospy.Rate(100)

        while not rospy.is_shutdown(): 
            try: 
                if  not isinstance( self.leader_joint_state , type(None) ) and \
                    not isinstance( self.follower_joint_state , type (None) ) :

                        self.apply_leader_position_to_follower()
                        self.recovery_follower_arm_error()
            except KeyboardInterrupt: 
                break


if __name__ == "__main__" : 

    telop = PandaToPandaTeleoperation()
