#! /usr/bin/env python3


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

    def __init__(self, leader_ns = "panda_leader", follower_ns = "panda_follower" ): 

        rospy.init_node("panda_panda_teleoperation") 

        self.leader_ns = leader_ns
        self.follower_ns = follower_ns
        

        self._teleoperation_variables()
        self._teleoperation_subscribers() 
        self._teleoperation_publishers() 
        # self._teleoperation_franka_gripper_homing()

        self.virtual_wall = { 
            "x" : { "max" : 0.65 , "min" : 0.2},
            "y" : { "max" : 0.37 , "min" : -0.43},
            "z" : { "max" : 0.58 , "min" : 0.05},
        }
        self.control_loop()

    def _teleoperation_variables(self): 

        self.leader_task_state = None
        self.follower_task_state = None

        self.leader_joint_state = None
        self.follower_joint_state = None

        self.leader_gripper_state = None
        self.leader_previous_gripper_state = None
        self.follower_gripper_state = None

        self.follower_joint_controller_pub = None
        self.leader_compliance_controller = None
        self.leader_compliance_config = None
        
    def _teleoperation_subscribers(self): 

        rospy.loginfo(self.leader_ns + "/" + self.leader_ns + "_state_controller/franka_states")
        rospy.Subscriber( self.leader_ns + "/joint_states" , JointState, self.leader_joint_state_cb )
        rospy.Subscriber( self.follower_ns + "/joint_states" , JointState, self.follower_joint_state_cb )
        rospy.Subscriber( self.leader_ns + "/" + self.leader_ns + "_state_controller/franka_states", FrankaState, self.leader_task_space_cb )
        rospy.Subscriber( self.follower_ns + "/" + self.follower_ns + "_state_controller/franka_states", FrankaState, self.follower_task_space_cb )

        rospy.Subscriber( self.leader_ns + "/franka_gripper/joint_states" , JointState , self.leader_gripper_state_cb )
        rospy.Subscriber( self.follower_ns + "/franka_gripper/joint_states" , JointState , self.follower_gripper_state_cb )

    def _teleoperation_publishers(self): 

        self.follower_joint_controller_pub = rospy.Publisher( self.follower_ns + "/" + self.follower_ns + "_joint_trajectory_controller/command" , JointTrajectory, queue_size = 1 )

        self.follower_joint_controller_client = actionlib.SimpleActionClient( self.follower_ns + "/" + self.follower_ns + "_joint_trajectory_controller/follow_joint_trajectory",   control_msgs.msg.FollowJointTrajectoryAction )
        self.follower_joint_controller_client.wait_for_server()

        self.follower_gripper_joint_controller_pub_move = actionlib.SimpleActionClient('/panda_follower/franka_gripper/move', franka_gripper.msg.MoveAction)
        self.follower_gripper_joint_controller_pub_grasp = actionlib.SimpleActionClient('/panda_follower/franka_gripper/grasp', franka_gripper.msg.GraspAction)
        self.follower_gripper_joint_controller_pub_stop = actionlib.SimpleActionClient('/panda_follower/franka_gripper/stop', franka_gripper.msg.StopAction)

        self.follower_gripper_joint_controller_pub_move.wait_for_server()
        self.follower_gripper_joint_controller_pub_stop.wait_for_server()
        

        self.follower_arm_recovery_client = actionlib.SimpleActionClient( "/panda_follower/franka_control/error_recovery", ErrorRecoveryAction )
        self.follower_arm_recovery_client.wait_for_server()

        self.leader_compliance_controller = dynamic_reconfigure.client.Client(self.leader_ns + "/" + self.leader_ns + "_cartesian_impedance_controller_controller_params", timeout=30, config_callback=self.leader_compliance_controller_cb)

    def _teleoperation_franka_gripper_homing(self): 
        leader_gripper = actionlib.SimpleActionClient('/panda_leader/franka_gripper/homing', franka_gripper.msg.HomingAction)
        follower_gripper = actionlib.SimpleActionClient('/panda_follower/franka_gripper/homing', franka_gripper.msg.HomingAction)

        leader_gripper.wait_for_server()
        follower_gripper.wait_for_server()

        # Creates a goal to send to the action server.
        goal = franka_gripper.msg.HomingGoal()

        # Sends the goal to the action server.
        leader_gripper.send_goal(goal)
        follower_gripper.send_goal(goal)

        # Waits for the server to finish performing the action.
        leader_gripper.wait_for_result()
        follower_gripper.wait_for_result()

        rospy.loginfo("Leader gripper homing state: {}".format(leader_gripper.get_result().success))
        rospy.loginfo("Follower gripper homing state: {}".format(follower_gripper.get_result().success))

    def leader_joint_state_cb (self, msg ): 
        self.leader_joint_state = msg 

    def leader_task_space_cb (self, msg ): 
        self.leader_task_state = msg 

    def follower_joint_state_cb (self, msg ): 
        self.follower_joint_state = msg 

    def follower_task_space_cb (self, msg ): 
        self.follower_task_state = msg 

    def leader_compliance_controller_cb(self, config):  
        self.leader_compliance_config = config

    def leader_gripper_state_cb (self, msg ): 
        self.leader_gripper_state = msg 

    def follower_gripper_state_cb(self, config):  
        self.follower_gripper_state = config

    def apply_leader_position_to_follower_action(self):


        trajectory = self.create_joint_trajectory()
        
        trajectory_goal = control_msgs.msg.FollowJointTrajectoryGoal()
        trajectory_goal.trajectory = trajectory

        self.follower_joint_controller_client.send_goal( trajectory_goal )
        self.follower_joint_controller_client.wait_for_result() 
        result = self.follower_joint_controller_client.get_result() 
        
        self.apply_virtual_walls()
 
    def create_joint_trajectory(self): 

        traj_point = JointTrajectoryPoint()
        trajectory = JointTrajectory()
        
        follower_joint_names = []
        Kp = 1.0
        Kd = 0.0

        for i in range(7):
            follower_joint_name = self.follower_ns + "_joint" + str(i+1)
            follower_joint_names.append( follower_joint_name ) 

        leader_joint_values = self.leader_task_state.q 
        follower_joint_values = self.follower_task_state.q 
        
        position_error = [ i - j for i, j in zip(leader_joint_values, follower_joint_values ) ]
        velocity_error = self.leader_task_state.dq
    
        correction = [ Kp * p_err + Kd * v_err for p_err, v_err in zip( position_error, velocity_error ) ]
        traj_point.positions = [ i + j for i, j in zip( follower_joint_values , correction ) ]
        traj_point.velocities = velocity_error

        traj_point.time_from_start = rospy.Duration( 0.75 )
        trajectory.header.stamp = rospy.Time().now()
        trajectory.joint_names = follower_joint_names
        trajectory.points.append( traj_point )

        o_t_ee = np.array( self.leader_task_state.O_T_EE ).reshape(4,4)

        return trajectory

    def apply_leader_position_to_follower(self): 
        
        trajectory = self.create_joint_trajectory()
        self.follower_joint_controller_pub.publish( trajectory )
        self.apply_virtual_walls()
 
    def apply_virtual_walls(self):
        
        max_force = 7.5
        o_t_ee = np.array( self.leader_task_state.O_T_EE ).reshape(4,4)
        x, y, z, _ = o_t_ee[3, :]

        if x > self.virtual_wall["x"]["max"]: 
            self.leader_compliance_config['task_haptic_x_force'] = -max_force
        elif x < self.virtual_wall["x"]["min"]: 
            self.leader_compliance_config['task_haptic_x_force'] = max_force
        else : 
            self.leader_compliance_config['task_haptic_x_force'] = 0

        if y > self.virtual_wall["y"]["max"]: 
            self.leader_compliance_config['task_haptic_y_force'] = -max_force
        elif y < self.virtual_wall["y"]["min"]: 
            self.leader_compliance_config['task_haptic_y_force'] = max_force
        else: 
            self.leader_compliance_config['task_haptic_y_force'] = 0

        if z > self.virtual_wall["z"]["max"]: 
            self.leader_compliance_config['task_haptic_z_force'] = -max_force
        elif z < self.virtual_wall["z"]["min"]: 
            self.leader_compliance_config['task_haptic_z_force'] = max_force
        else: 
            self.leader_compliance_config['task_haptic_z_force'] = 0

        try:
            self.leader_compliance_controller.update_configuration( self.leader_compliance_config )
        except: 
            rospy.logwarn( "Error in cleanly exiting the teleop part" )

    def apply_leader_gripper_position_to_follower(self):
        
        # stop the last command
        stop_goal = franka_gripper.msg.StopAction()
        self.follower_gripper_joint_controller_pub_stop.send_goal( stop_goal )
 
        # send a new goal
        goal = franka_gripper.msg.MoveGoal()
        goal.width = self.leader_gripper_state.position[0] * 2
        goal.speed = 0.1
        # goal.force = 30
        # goal.epsilon.inner = 0.2
        # goal.epsilon.outer = 0.2

        # self.follower_gripper_joint_controller_pub_move.cancel_goal()
        self.follower_gripper_joint_controller_pub_move.send_goal(goal)
        self.follower_gripper_joint_controller_pub_move.wait_for_result()

        result = self.follower_gripper_joint_controller_pub_move.get_result()
        if not isinstance(result , type (None)): 
            if not result.success: 
                rospy.logerr("Error in follower gripper move teleop: {}".format(result.error))


    def grasp_object(self, width): 

        # send a new goal
        goal = franka_gripper.msg.GraspGoal()
        goal.width = width
        goal.speed = 0.05
        goal.force = 30
        goal.epsilon.inner = 0.2
        goal.epsilon.outer = 0.2

        self.follower_gripper_joint_controller_pub_grasp.send_goal(goal)
        self.follower_gripper_joint_controller_pub_grasp.wait_for_result()

        result = self.follower_gripper_joint_controller_pub_grasp.get_result()
        if not isinstance(result , type (None)): 
            if not result.success: 
                rospy.logerr("Error in follower gripper grasp teleop: {}".format(result.error))

    def open_gripper(self): 

        # send a new goal
        goal = franka_gripper.msg.MoveGoal()
        goal.width = 0.075
        goal.speed = 0.05

        self.follower_gripper_joint_controller_pub_move.send_goal(goal)
        self.follower_gripper_joint_controller_pub_move.wait_for_result()

        result = self.follower_gripper_joint_controller_pub_move.get_result()
        if not isinstance(result , type (None)): 
            if not result.success: 
                rospy.logerr("Error in follower gripper move teleop: {}".format(result.error))


    def recovery_follower_arm_error(self): 

        goal = ErrorRecoveryActionGoal() 

        self.follower_arm_recovery_client.send_goal(goal)
        # self.follower_arm_recovery_client.wait_for_result() 

    def create_instruction_window(self): 

        w, h = 320, 620
        image = np.zeros((w,h) , dtype = np.uint8) 

        # write text here 

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = ( 40 , 20 )
        fontScale              = 0.75
        fontColor              = (255,255,255)
        thickness              = 1
        lineType               = 2

        cv2.putText(image,'Press "q" to close the gripper', 
            ( 50 , 50 ), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        cv2.putText(image,'Press "w" to close the gripper', 
            ( 50 , 150 ), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        cv2.putText(image,'Stack the three cubes on top of each other', 
            ( 50 , 250 ), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        return image 


    def control_loop(self): 
        
        move_gripper_to_leader_gripper = False

        rospy.loginfo("Press Ctrl+C if anything goes sideways")
        rate = rospy.Rate(100)

        while not rospy.is_shutdown(): 
            try: 
                if  not isinstance( self.leader_joint_state , type(None) ) and \
                    not isinstance( self.leader_task_state, type(None) ) and \
                    not isinstance( self.leader_gripper_state , type (None) ) and \
                    not isinstance( self.leader_previous_gripper_state , type (None) ) and \
                    not isinstance( self.follower_joint_state , type (None) ) and \
                    not isinstance( self.follower_task_state , type (None) ) and \
                    not isinstance( self.follower_gripper_state , type (None) ) and \
                    not isinstance( self.leader_compliance_config, type (None) ): 

                        self.apply_leader_position_to_follower()
                        # self.apply_leader_gripper_position_to_follower() 
                        self.recovery_follower_arm_error()

                        # image = self.create_instruction_window() 
                        # cv2.imshow("Instructions" , image ) 
                        # k = cv2.waitKey(30) 
                        # if k == 27: 
                        #     cv2.destroyAllWindows()
                        #     break 
                        # elif k  == -1: 
                        #     pass
                        # else: 
                        #     if k == ord('q') : 
                        #         self.grasp_object( 0.04 ) # small cube
                            
                        #     if k == ord('w'): 
                        #         self.open_gripper() # open gripper


                elif not isinstance( self.leader_gripper_state , type (None) ) and isinstance( self.leader_previous_gripper_state , type (None) ):

                        self.leader_previous_gripper_state = self.leader_gripper_state

            except KeyboardInterrupt: 
                break


if __name__ == "__main__" : 

    telop = PandaToPandaTeleoperation()
