#!/usr/bin/python3
import sys

import mcr_states.common.basic_states as gbs
import mir_states.common.manipulation_states as gms
import rospy
import smach
from mir_planning_msgs.msg import (
    GenericExecuteAction,
    GenericExecuteFeedback,
    GenericExecuteResult,
)
from geometry_msgs.msg import PoseStamped
import smach.state
from mir_actions.utils import Utils
from smach_ros import ActionServerWrapper, IntrospectionServer
import numpy as np
import tf.transformations as tr
from brics_actuator.msg import JointPositions, JointValue
from sensor_msgs.msg import JointState
import numpy as np
import tf
from std_msgs.msg import String

# Global variable to store the latest pose from the topic
latest_empty_space_pose = None

class MoveArmUp(smach.State):

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "timeout"],
        )
        self.joint_states_sub = rospy.Subscriber("/joint_states", JointState, self.joint_states_cb)
        self.pub_arm_position = rospy.Publisher("/arm_1/arm_controller/position_command", JointPositions, queue_size=1)
        self.current_joint_positions = None
        self.is_arm_moving = False
        self.zero_vel_counter = 0
        self.joint_1_position = 1.3787 #1.8787

    def joint_states_cb(self, msg):
        if "arm_joint_1" in msg.name: # get the joint values of the arm only
            self.current_joint_positions = msg.position

        self.joint_state = msg
        # monitor the velocities
        self.joint_velocities = msg.velocity
        # if all velocities are 0.0, the arm is not moving
        if "arm_joint_1" in msg.name and all([v == 0.0 for v in self.joint_velocities]):
            self.zero_vel_counter += 1

    def execute(self, userdata):
        self.current_joint_positions = None
        while not rospy.is_shutdown():
            rospy.sleep(0.1)
            if self.current_joint_positions is not None:
                break
        joint_values = self.current_joint_positions[:]
        joint_values = list(joint_values)
        joint_values[1] -= 0.3 # self.joint_1_position
        
        names = self.joint_state.name

        joint_positions = JointPositions()
        joint_positions.positions = [
            JointValue(
                rospy.Time.now(),
                joint_name,
                "rad",
                joint_value
            )
            for joint_name, joint_value in zip(names, joint_values)
        ]
        self.pub_arm_position.publish(joint_positions)
        rospy.sleep(1)
        return "succeeded"

# ===============================================================================


class TriggerEmptySpaceDetection(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'])
        self.trigger_pub = rospy.Publisher('/empty_space_detector/event_in', String, queue_size=10)
        self.event_sub = rospy.Subscriber('/empty_space_detector/event_out', String, self.event_callback)
        self.event_received = False

    def event_callback(self, msg):
        if msg.data == "e_empty_space_detected":
            self.event_received = True

    def execute(self, userdata):
        self.event_received = False
        self.trigger_pub.publish(String("e_empty"))
        
        timeout = rospy.Duration(10.0)  # 10 seconds timeout
        start_time = rospy.Time.now()
        
        while not self.event_received and (rospy.Time.now() - start_time) < timeout:
            rospy.sleep(0.1)
        
        if self.event_received:
            return 'succeeded'
        else:
            return 'failed'

class TriggerPointCloudProcessing(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'])
        self.trigger_pub = rospy.Publisher('/empty_space_detector/event_in', String, queue_size=10)
        self.event_sub = rospy.Subscriber('/empty_space_detector/event_out', String, self.event_callback)
        self.event_received = False

    def event_callback(self, msg):
        if msg.data == "e_pointcloud_processed":
            self.event_received = True

    def execute(self, userdata):
        self.event_received = False
        self.trigger_pub.publish(String("e_cloud"))
        
        timeout = rospy.Duration(10.0)  # 10 seconds timeout
        start_time = rospy.Time.now()
        
        while not self.event_received and (rospy.Time.now() - start_time) < timeout:
            rospy.sleep(0.1)
        
        if self.event_received:
            return 'succeeded'
        else:
            return 'failed'

class SendStopEvent(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'])
        self.stop_pub = rospy.Publisher('/empty_space_detector/event_in', String, queue_size=10)

    def execute(self, userdata):
        self.stop_pub.publish(String("e_stop"))
        rospy.sleep(0.1)  # Small delay to ensure the message is sent
        return 'succeeded'


class PublishEmptyspacePose(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["success", "failed"],
                                    input_keys=["goal"],
                                    output_keys=["move_arm_to"])

        self.empty_space_pose = rospy.Publisher(
            "mcr_perception/object_selector/output/object_pose",
            PoseStamped,
            queue_size=10)
    

    def pose_callback(self, msg):
        self.received_pose = msg

    def execute(self, userdata):
        global latest_empty_space_pose

        self.received_pose = None
        sub = rospy.Subscriber("/empty_space_pose", PoseStamped, self.pose_callback)

        timeout = rospy.Duration(5.0)  # 5 seconds timeout
        start_time = rospy.Time.now()

        while self.received_pose is None and (rospy.Time.now() - start_time) < timeout:
            rospy.sleep(0.1)
        
        sub.unregister() # Unsubscribe from the topic after receiving the pose

        latest_empty_space_pose = self.received_pose

        
        if latest_empty_space_pose is None:
            rospy.logerr("No pose received from /empty_space_pose topic")
            return "failed"

        # Create a new PoseStamped message
        modified_pose = PoseStamped()
        
        # Copy the header and position from the latest pose
        modified_pose.header.frame_id = "base_link"
        modified_pose.pose.position = latest_empty_space_pose.pose.position
        
        # Copy the orientation, but set w to 1.0
        # modified_pose.pose.orientation.x = latest_empty_space_pose.pose.orientation.x
        # modified_pose.pose.orientation.y = latest_empty_space_pose.pose.orientation.y
        # modified_pose.pose.orientation.z = latest_empty_space_pose.pose.orientation.z
        # modified_pose.pose.orientation.w = 1.0

        modified_pose.pose.orientation.x = 0.0
        modified_pose.pose.orientation.y = 0.0
        modified_pose.pose.orientation.z = 0.0
        modified_pose.pose.orientation.w = 1.0

        self.empty_space_pose.publish(modified_pose)

        # self.empty_space_pose.publish(latest_empty_space_pose)
        return "success"

def main():
    # Open the container
    rospy.init_node("mir_cartesian_test_server")


    # Construct state machine
    sm = smach.StateMachine(
        outcomes=["OVERALL_SUCCESS", "OVERALL_FAILED"],
        input_keys=["goal"],
        output_keys=["feedback", "result"],
    )

    # Initialize feedback and result in userdata
    sm.userdata.feedback = GenericExecuteFeedback()
    sm.userdata.result = GenericExecuteResult()

    with sm:
        smach.StateMachine.add(
            "CLOSE_GRIPPER",
            gms.control_gripper("close"),
            transitions={"succeeded": "TRIGGER_EMPTY_SPACE_DETECTION", 
                         "timeout": "OVERALL_FAILED"},
        )

        smach.StateMachine.add(
            "TRIGGER_EMPTY_SPACE_DETECTION",
            TriggerEmptySpaceDetection(),
            transitions={
                "succeeded": "TRIGGER_POINT_CLOUD_PROCESSING",
                "failed": "OVERALL_FAILED",
            },
        )

        smach.StateMachine.add(
            "TRIGGER_POINT_CLOUD_PROCESSING",
            TriggerPointCloudProcessing(),
            transitions={
                "succeeded": "MOVE_ARM_TO_PRE_PLACE",
                "failed": "OVERALL_FAILED",
            },
        )

        smach.StateMachine.add(
            "MOVE_ARM_TO_PRE_PLACE",
            gms.move_arm("pre_place", use_moveit=False),
            transitions={
                "succeeded": "PUBLISH_OBJECT_POSE",
                "failed": "OVERALL_FAILED",
            },
        )


        smach.StateMachine.add(
            "PUBLISH_OBJECT_POSE",
            PublishEmptyspacePose(),
            transitions={
                "success": "CHECK_PICK_POSE_IK",
                "failed": "PUBLISH_OBJECT_POSE",
            },
        )
        smach.StateMachine.add(
            "CHECK_PICK_POSE_IK",
            gbs.send_and_wait_events_combined(
                event_in_list=[
                    ("/pregrasp_planner_node/event_in", "e_start")
                ],
                event_out_list=[
                    (
                        "/pregrasp_planner_node/event_out",
                        "e_success",
                        True,
                    )
                ],
                timeout_duration=20,
            ),
            transitions={
                "success": "GO_TO_PICK_POSE",
                "timeout": "CHECK_PICK_POSE_IK", 
                "failure": "OVERALL_FAILED",
            },
        )

        smach.StateMachine.add(
            "GO_TO_PICK_POSE",
            gbs.send_and_wait_events_combined(
                event_in_list=[
                    ("/waypoint_trajectory_generation/event_in", "e_start")],
                event_out_list=[
                    (
                        "/waypoint_trajectory_generation/event_out",
                        "e_success",
                        True,
                    )],
                timeout_duration=20,
            ),
            transitions={
                "success": "RELEASE_GRIPPER", 
                "timeout": "OVERALL_FAILED",
                "failure": "OVERALL_FAILED",
            },
        )


        smach.StateMachine.add(
                "RELEASE_GRIPPER",
                gms.control_gripper('release'),
                transitions={
			        "succeeded": "SEND_STOP_EVENT",
                         "timeout": "MOVE_ARM_UP"}
        )

        smach.StateMachine.add(
            "SEND_STOP_EVENT",
            SendStopEvent(),
            transitions={
                "succeeded": "MOVE_ARM_UP",
            },
        )

        smach.StateMachine.add(
                "MOVE_ARM_UP",
                MoveArmUp(),
                transitions={
			        "succeeded": "MOVE_ARM_TO_NEUTRAL",
                                 "timeout": "MOVE_ARM_TO_NEUTRAL"}
        )

        smach.StateMachine.add(
                "MOVE_ARM_TO_NEUTRAL",
                gms.move_arm("pre_place", use_moveit=False),
                transitions={
                    "succeeded": "OPEN_GRIPPER",
                    "failed": "MOVE_ARM_TO_NEUTRAL",
            },
        )

        smach.StateMachine.add(
            "OPEN_GRIPPER",
            gms.control_gripper("open"),
            transitions={
                "succeeded": "OVERALL_SUCCESS",
                "timeout": "OVERALL_FAILED",
            },
        )

    # Construct action server wrapper
    asw = ActionServerWrapper(
        server_name="mir_cartesian_test_server",
        action_spec=GenericExecuteAction,
        wrapped_container=sm,
        succeeded_outcomes=["OVERALL_SUCCESS"],
        aborted_outcomes=["OVERALL_FAILED"],
        preempted_outcomes=["PREEMPTED"],
        goal_key="goal",
        feedback_key="feedback",
        result_key="result",
    )

    # Run the server in a background thread
    asw.run_server()
    rospy.spin()

if __name__ == "__main__":
    main()
