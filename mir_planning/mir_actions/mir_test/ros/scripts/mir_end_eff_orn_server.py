#!/usr/bin/python3

import sys

from selectors import PollSelector
import mcr_states.common.basic_states as gbs
import mir_states.common.manipulation_states as gms
import mir_states.common.action_states as gas
import rospy
import smach
from mir_actions.utils import Utils
from mir_planning_msgs.msg import (
    GenericExecuteAction,
    GenericExecuteFeedback,
    GenericExecuteResult,
    GenericExecuteGoal
)
from smach_ros import ActionServerWrapper, IntrospectionServer
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, PoseStamped
from diagnostic_msgs.msg import KeyValue
from actionlib import SimpleActionClient
from actionlib_msgs.msg import GoalStatus
from brics_actuator.msg import JointPositions, JointValue
from sensor_msgs.msg import JointState
import numpy as np
import tf

import tf2_ros
import tf2_geometry_msgs

# Removed duplicate and unused imports


class PublishObjectpose(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                             outcomes=["success", "failed"],
                             input_keys=["goal"],
                             output_keys=["move_arm_to"])
        self.object_pose_pub = rospy.Publisher(
            "mcr_perception/object_selector/output/object_pose",
            PoseStamped,
            queue_size=10
        )

    def execute(self, userdata):
        try:
            # Create a PoseStamped message with hardcoded x, y, z values
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "base_link_static"  # Set the appropriate frame_id

            # Set the hardcoded position values
            pose_msg.pose.position.x = 0.5708119916915894
            pose_msg.pose.position.y = -0.24560791528224946  
            pose_msg.pose.position.z =  0.014999999664723873

            # Set orientation (no rotation)
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.009567914557666314
            pose_msg.pose.orientation.w =  0.9999542264579

            # Publish the pose
            self.object_pose_pub.publish(pose_msg)
            rospy.loginfo("Published hardcoded pose to mcr_perception/object_selector/output/object_pose")
            return "success"

        except Exception as e:
            rospy.logerr("Failed to publish hardcoded pose: %s" % str(e))
            return "failed"


class MoveArmUp(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "timeout"])
        self.joint_states_sub = rospy.Subscriber("/joint_states", JointState, self.joint_states_cb)
        self.pub_arm_position = rospy.Publisher("/arm_1/arm_controller/position_command", JointPositions, queue_size=1)
        self.current_joint_positions = None

    def joint_states_cb(self, msg):
        if "arm_joint_1" in msg.name:
            self.current_joint_positions = msg.position

    def execute(self, userdata):
        self.current_joint_positions = None
        while not rospy.is_shutdown() and self.current_joint_positions is None:
            rospy.sleep(0.1)
        joint_values = list(self.current_joint_positions)
        joint_values[1] -= 0.3
        joint_positions = JointPositions()
        joint_positions.positions = [
            JointValue(rospy.Time.now(), name, "rad", value)
            for name, value in zip(["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"], joint_values)
        ]
        self.pub_arm_position.publish(joint_positions)
        rospy.sleep(1)
        return "succeeded"


def main():
    rospy.init_node("mir_cartesian_test_server")
    sm = smach.StateMachine(outcomes=["OVERALL_SUCCESS", "OVERALL_FAILED"],
                            input_keys=["goal"],
                            output_keys=["feedback", "result"])
    sm.userdata.feedback = GenericExecuteFeedback()
    sm.userdata.result = GenericExecuteResult()

    with sm:
        smach.StateMachine.add(
            "SET_PREGRASP_PARAMS",
            gbs.set_named_config("pregrasp_planner_no_sampling"),
            transitions={
                "success": "MOVE_ARM_TO_PRE_PLACE",
                "timeout": "OVERALL_FAILED",
                "failure": "OVERALL_FAILED",
            },
        )

        
        smach.StateMachine.add(
            "MOVE_ARM_TO_PRE_PLACE",
            gms.move_arm("pre_place_m20", use_moveit=False),
            transitions={
                "succeeded": "PUBLISH_OBJECT_POSE",
                "failed": "OVERALL_FAILED",
            },
        )

        smach.StateMachine.add(
            "PUBLISH_OBJECT_POSE",
            PublishObjectpose(),
            transitions={
                "success": "SET_DBC_PARAMS",
                "failed": "OVERALL_FAILED"
            },
        )

        smach.StateMachine.add(
            "SET_DBC_PARAMS",
            gbs.set_named_config("dbc_pick_object"),
            transitions={
                "success": "MOVE_ROBOT_AND_TRY_INSERTING",
                "timeout": "OVERALL_FAILED",
                "failure": "OVERALL_FAILED",
            },
        )

        smach.StateMachine.add(
            "MOVE_ROBOT_AND_TRY_INSERTING",
            gbs.send_and_wait_events_combined(
                event_in_list=[("/wbc/event_in", "e_start")],
                event_out_list=[("/wbc/event_out", "e_success", True)],
                timeout_duration=50,
            ),
            transitions={
                "success": "OPEN_GRIPPER",
                "timeout": "STOP_MOVE_ROBOT_TO_OBJECT_WITH_FAILURE",
                "failure": "STOP_MOVE_ROBOT_TO_OBJECT_WITH_FAILURE",
            },
        )

        smach.StateMachine.add(
            "STOP_MOVE_ROBOT_TO_OBJECT_WITH_FAILURE",
            gbs.send_event([
                ("/waypoint_trajectory_generation/event_in", "e_stop"),
                ("/wbc/event_in", "e_stop"),
            ]),
            transitions={"success": "OVERALL_FAILED"},
        )

        smach.StateMachine.add(
            "OPEN_GRIPPER",
            gms.control_gripper('open'),
            transitions={
                "succeeded": "MOVE_ARM_UP",
                "timeout": "MOVE_ARM_UP",
            },
        )

        smach.StateMachine.add(
            "MOVE_ARM_UP",
            MoveArmUp(),
            transitions={
                "succeeded": "MOVE_ARM_TO_HOLD",
                "timeout": "MOVE_ARM_UP",
            },
        )

        smach.StateMachine.add(
            "MOVE_ARM_TO_HOLD",
            gms.move_arm("pre_place"),
            transitions={
                "succeeded": "OVERALL_SUCCESS",
                "failed": "OVERALL_FAILED",
            },
        )

    asw = ActionServerWrapper(
        server_name="mir_end_orn_server",
        action_spec=GenericExecuteAction,
        wrapped_container=sm,
        succeeded_outcomes=["OVERALL_SUCCESS"],
        aborted_outcomes=["OVERALL_FAILED"],
        preempted_outcomes=["PREEMPTED"],
        goal_key="goal",
        feedback_key="feedback",
        result_key="result",
    )
    asw.run_server()
    rospy.spin()


if __name__ == "__main__":
    main()
