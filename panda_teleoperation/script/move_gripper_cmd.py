#! /usr/bin/env python3
import franka_gripper.msg
import rospy
import sys

# Brings in the SimpleActionClient
import actionlib


# Brings in the messages used by the grasp action, including the
# goal message and the result message.

def grasp_client():
    # Creates the SimpleActionClient, passing the type of the action
    # (GraspAction) to the constructor.
    client = actionlib.SimpleActionClient('/panda_follower/franka_gripper/move', franka_gripper.msg.MoveAction)

    # Waits until the action server has started up and started
    # listening for goals.
    client.wait_for_server()

    # Creates a goal to send to the action server.
    goal = franka_gripper.msg.GraspGoal()
    goal.width = 0.04
    goal.speed = 0.1
    # goal.force = 5
    # goal.epsilon.inner = 0.005
    # goal.epsilon.outer = 0.005


    # Sends the goal to the action server.
    client.send_goal(goal)

    # Waits for the server to finish performing the action.
    client.wait_for_result()

    # Prints out the result of executing the action
    return client.get_result()  # A GraspResult


if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        rospy.init_node('grasp_client_py')
        result = grasp_client()
        print("Success: ",result.success)
        print("Error message: ", result.error)
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)