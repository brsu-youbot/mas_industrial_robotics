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

class MoveDBCPose(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded"])
        self._dbc_pose_pub = rospy.Publisher(
            "/mcr_navigation/direct_base_controller/input_pose",
            PoseStamped,
            queue_size=1,
        )
        self.last_pose = None
        self.sub = rospy.Subscriber("/wbc/base_motion_pose", PoseStamped, self.pose_cb)
        self.listener = tf.TransformListener()  # INITIALIZE LISTENER

    def pose_cb(self, msg):
        self.last_pose = msg

    def execute(self, userdata):
        try:
            # Get pose from WBC
            if self.last_pose is None:
                rospy.logwarn("No cached WBC pose, waiting...")
                incoming_pose = rospy.wait_for_message("/wbc/base_motion_pose", PoseStamped, timeout=rospy.Duration(1.0))
            else:
                incoming_pose = self.last_pose

            # Get current base orientation from TF
            tf_msg = self.listener.lookupTransform("/base_link_static", "/base_link", rospy.Time(0))

            # Build modified pose
            modified_pose = PoseStamped()
            modified_pose.header.frame_id = "base_link_static"  # CORRECT HEADER
            modified_pose.header.stamp = rospy.Time.now()
            modified_pose.pose.position.x =tf_msg[0][0] + (-incoming_pose.pose.position.x)
            modified_pose.pose.position.y =tf_msg[0][1] + (-incoming_pose.pose.position.y)
            modified_pose.pose.position.z = tf_msg[0][2]
            modified_pose.pose.orientation.x = tf_msg[1][0]
            modified_pose.pose.orientation.y = tf_msg[1][1]
            modified_pose.pose.orientation.z = tf_msg[1][2]
            modified_pose.pose.orientation.w = tf_msg[1][3]

            self._dbc_pose_pub.publish(modified_pose)
            return "succeeded"

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("TF error in MoveDBC: %s", str(e))
            return "succeeded"
        except rospy.ROSException as e:
            rospy.logerr("ROS error in MoveDBC: %s", str(e))
            return "succeeded"


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
        timeout = rospy.Duration(10.0)
        start_time = rospy.Time.now()
        while not self.event_received and (rospy.Time.now() - start_time) < timeout:
            rospy.sleep(0.1)
        return 'succeeded' if self.event_received else 'failed'

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
        timeout = rospy.Duration(10.0)
        start_time = rospy.Time.now()
        while not self.event_received and (rospy.Time.now() - start_time) < timeout:
            rospy.sleep(0.1)
        return 'succeeded' if self.event_received else 'failed'

class SendStopEvent(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'])
        self.stop_pub = rospy.Publisher('/empty_space_detector/event_in', String, queue_size=10)

    def execute(self, userdata):
        self.stop_pub.publish(String("e_stop"))
        rospy.sleep(0.1)
        return 'succeeded'

class PublishEmptyspacePosetoBaseStatic(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["nor_success", "wbc_success", "failed"],
                                    input_keys=["goal"],
                                    output_keys=["move_arm_to"])

        self.empty_space_pose_pub = rospy.Publisher(
            "mcr_perception/object_selector/output/object_pose",
            PoseStamped,
            queue_size=10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def pose_callback(self, msg):
        self.received_pose = msg

    def execute(self, userdata):
        self.received_pose = None
        sub = rospy.Subscriber("/empty_space_pose", PoseStamped, self.pose_callback)
        
        timeout = rospy.Duration(5.0)  # 5 seconds timeout
        start_time = rospy.Time.now()
        
        while self.received_pose is None and (rospy.Time.now() - start_time) < timeout:
            rospy.sleep(0.1)
        
        sub.unregister()  # Unsubscribe from the topic after receiving the pose
        
        if self.received_pose is None:
            rospy.logerr("No pose received from /empty_space_pose topic")
            return "failed"
        
        try:
            # Lookup transform from the pose's frame to base_static_link
            transform = self.tf_buffer.lookup_transform(
                "base_link_static",  # Target frame
                self.received_pose.header.frame_id,  # Source frame
                rospy.Time(0),  # Get the latest available transform
                rospy.Duration(1.0)  # Timeout for transform availability
            )
            
            # Transform pose to base_static_link frame
            transformed_pose = tf2_geometry_msgs.do_transform_pose(self.received_pose, transform)

            
            # Set the orientation to a "straight" predefined quaternion
            transformed_pose.pose.orientation.x = 0.0
            transformed_pose.pose.orientation.y = 0.0
            transformed_pose.pose.orientation.z = 0.0
            transformed_pose.pose.orientation.w = 1.0  # Identity quaternion

            rospy.loginfo("Transformed pose: %s" % transformed_pose)
            
            # Publish the transformed pose
            self.empty_space_pose_pub.publish(transformed_pose)
            
            # Check the y value of the transformed pose
            y_value = transformed_pose.pose.position.y
            
            if -0.05 <= y_value <= 0.1:
                return "nor_success"
            else:
                return "wbc_success"
        
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
            rospy.logerr("Failed to transform pose: %s" % str(e))
            return "failed"



def main():
    rospy.init_node("mir_cartesian_test_server")
    sm = smach.StateMachine(
        outcomes=["OVERALL_SUCCESS", "OVERALL_FAILED"],
        input_keys=["goal"],
        output_keys=["feedback", "result"],
    )
    sm.userdata.feedback = GenericExecuteFeedback()
    sm.userdata.result = GenericExecuteResult()

    with sm:
        smach.StateMachine.add(
            "SET_PREGRASP_PARAMS",
            gbs.set_named_config("pregrasp_planner_no_sampling"),
            transitions={
                "success": "CLOSE_GRIPPER",
                "timeout": "OVERALL_FAILED",
                "failure": "OVERALL_FAILED",
            },
        )

        smach.StateMachine.add(
            "CLOSE_GRIPPER",
            gms.control_gripper("close"),
            transitions={"succeeded": "TRIGGER_EMPTY_SPACE_DETECTION", "timeout": "OVERALL_FAILED"},
        )

        smach.StateMachine.add(
            "TRIGGER_EMPTY_SPACE_DETECTION",
            TriggerEmptySpaceDetection(),
            transitions={"succeeded": "TRIGGER_POINT_CLOUD_PROCESSING", "failed": "OVERALL_FAILED"},
        )

        smach.StateMachine.add(
            "TRIGGER_POINT_CLOUD_PROCESSING",
            TriggerPointCloudProcessing(),
            transitions={"succeeded": "PUBLISH_REFERENCE_FRAME", "failed": "OVERALL_FAILED"},
        )

        smach.StateMachine.add(
            "PUBLISH_REFERENCE_FRAME",
            gbs.send_event([("/static_transform_publisher_node/event_in", "e_start")]),
            transitions={"success": "MOVE_ARM_TO_PRE_PLACE"},
        )

        smach.StateMachine.add(
            "MOVE_ARM_TO_PRE_PLACE",
            gms.move_arm("pre_place", use_moveit=False),
            transitions={
                # "succeeded": "SET_DBC_PARAMS", 
                "succeeded": "PUBLISH_OBJECT_POSE_AS_STATIC",
                "failed": "OVERALL_FAILED"},
        )

        
        smach.StateMachine.add(
            "PUBLISH_OBJECT_POSE_AS_STATIC",
            PublishEmptyspacePosetoBaseStatic(),
            transitions={ 
                "wbc_success": "SET_DBC_PARAMS",
                "nor_success": "CHECK_PICK_POSE_IK",
                "failed": "OVERALL_FAILED"
            },
        )

        # WBC

        smach.StateMachine.add(
            "SET_DBC_PARAMS",
            gbs.set_named_config("dbc_pick_object"),
            transitions={
                "success": "SEND_STOP_EVENT_WBC",
                "timeout": "OVERALL_FAILED",
                "failure": "OVERALL_FAILED",
            },
        )

        smach.StateMachine.add(
            "SEND_STOP_EVENT_WBC",
            SendStopEvent(),
            transitions={"succeeded": "MOVE_ROBOT_AND_TRY_INSERTING"},
        )

        smach.StateMachine.add(
            "MOVE_ROBOT_AND_TRY_INSERTING",
            gbs.send_and_wait_events_combined(
                event_in_list=[("/wbc/event_in", "e_start")],
                event_out_list=[("/wbc/event_out", "e_success", True)],
                timeout_duration=50,
            ),
            transitions={
                "success": "RELEASE_GRIPPER",
                "timeout": "STOP_MOVE_ROBOT_TO_OBJECT_WITH_FAILURE",
                "failure": "STOP_MOVE_ROBOT_TO_OBJECT_WITH_FAILURE",
            },
        )

        smach.StateMachine.add(
            "STOP_MOVE_ROBOT_TO_OBJECT_WITH_FAILURE",
            gbs.send_event(
                [
                    ("/waypoint_trajectory_generation/event_in", "e_stop"),
                    ("/wbc/event_in", "e_stop"),
                ]
            ),
            transitions={"success": "OVERALL_FAILED"},
        )

        smach.StateMachine.add(
            "RELEASE_GRIPPER",
            gms.control_gripper('release'),
            transitions={"succeeded": "MOVE_ARM_UP", "timeout": "MOVE_ARM_UP"},
        )

        smach.StateMachine.add(
            "MOVE_ARM_UP",
            MoveArmUp(),
            transitions={"succeeded": "MOVE_ARM_TO_NEUTRAL", "timeout": "MOVE_ARM_TO_NEUTRAL"},
        )

        smach.StateMachine.add(
            "MOVE_ARM_TO_NEUTRAL",
            gms.move_arm("pre_place", use_moveit=False),
            transitions={"succeeded": "MOVE_DBC_IN_Y", "failed": "MOVE_ARM_TO_NEUTRAL"},
        )

        smach.StateMachine.add(
            "MOVE_DBC_IN_Y",
            MoveDBCPose(),
            transitions={"succeeded": "MOVE_BASE_USING_DBC"},
        )

        smach.StateMachine.add(
            "MOVE_BASE_USING_DBC",
            gbs.send_and_wait_events_combined(
                event_in_list=[("/mcr_navigation/direct_base_controller/coordinator/event_in", "e_start")],
                event_out_list=[("/mcr_navigation/direct_base_controller/coordinator/event_out", "e_success", True)],
                timeout_duration=10,
            ),
            transitions={"success": "OPEN_GRIPPER", "timeout": "OVERALL_FAILED", "failure": "OVERALL_FAILED"},
        )


        # IK

        smach.StateMachine.add(
            "CHECK_PICK_POSE_IK",
            gbs.send_and_wait_events_combined(
                event_in_list=[("/pregrasp_planner_node/event_in", "e_start")],
                event_out_list=[("/pregrasp_planner_node/event_out", "e_success", True)],
                timeout_duration=20,
            ),
            transitions={"success": "SEND_STOP_EVENT_NO_WBC", "timeout": "CHECK_PICK_POSE_IK", "failure": "OVERALL_FAILED"},
        )

        smach.StateMachine.add(
            "SEND_STOP_EVENT_NO_WBC",
            SendStopEvent(),
            transitions={"succeeded": "GO_TO_PICK_POSE"},
        )

        smach.StateMachine.add(
            "GO_TO_PICK_POSE",
            gbs.send_and_wait_events_combined(
                event_in_list=[("/waypoint_trajectory_generation/event_in", "e_start")],
                event_out_list=[("/waypoint_trajectory_generation/event_out", "e_success", True)],
                timeout_duration=20,
            ),
            transitions={"success": "RELEASE_GRIPPER_IK", "timeout": "OVERALL_FAILED", "failure": "OVERALL_FAILED"},
        )

        smach.StateMachine.add(
            "RELEASE_GRIPPER_IK",
            gms.control_gripper('release'),
            transitions={"succeeded": "MOVE_ARM_UP_IK", "timeout": "MOVE_ARM_UP_IK"},
        )

        smach.StateMachine.add(
            "MOVE_ARM_UP_IK",
            MoveArmUp(),
            transitions={"succeeded": "MOVE_ARM_TO_NEUTRAL_IK", "timeout": "MOVE_ARM_TO_NEUTRAL_IK"},
        )

        smach.StateMachine.add(
            "MOVE_ARM_TO_NEUTRAL_IK",
            gms.move_arm("pre_place", use_moveit=False),
            transitions={"succeeded": "OPEN_GRIPPER", "failed": "MOVE_ARM_TO_NEUTRAL_IK"},
        )

        smach.StateMachine.add(
            "OPEN_GRIPPER",
            gms.control_gripper("open"),
            transitions={"succeeded": "OVERALL_SUCCESS", "timeout": "OVERALL_FAILED"},
        )

        

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
    asw.run_server()
    rospy.spin()

if __name__ == "__main__":
    main()