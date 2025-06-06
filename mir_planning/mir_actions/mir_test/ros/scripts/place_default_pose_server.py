#!/usr/bin/env python

import rospy
import smach
import smach_ros
from std_msgs.msg import String
from mir_actions.utils import Utils
from mir_planning_msgs.msg import GenericExecuteAction, GenericExecuteFeedback, GenericExecuteResult
from smach_ros import ActionServerWrapper
import mir_states.common.manipulation_states as gms
import mcr_states.common.basic_states as gbs
import random

class DefaultSafePose(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "failed"],
                                    input_keys=["goal","move_arm_to"],
                                    output_keys=["move_arm_to"])

    def execute(self, userdata):
        rospy.logwarn("Checking pre-defined safe pose")
        location = Utils.get_value_of(userdata.goal.parameters, "location")
        current_platform_height = rospy.get_param("/"+location)
        
        random_pose_index = random.randint(1, 4)  # Select a pose randomly
        selected_pose = f"pose{random_pose_index}"
        
        userdata.move_arm_to = f"{current_platform_height}cm/{selected_pose}"
        rospy.loginfo(f"Selected pose: {userdata.move_arm_to}")
        
        rospy.sleep(0.1)
        return "succeeded"

class GetPoseToPlaceObject(smach.State):
    def __init__(self, topic_name_pub, topic_name_sub, event_sub, timeout_duration):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "failed"],
            input_keys=["goal"],
            output_keys=["move_arm_to"],
        )
        self.timeout = rospy.Duration.from_sec(timeout_duration)
        self.platform_name_pub = rospy.Publisher(topic_name_pub, String, queue_size=10)
        rospy.Subscriber(topic_name_sub, String, self.pose_cb)
        rospy.Subscriber(event_sub, String, self.event_cb)
        rospy.sleep(0.1)  # Time for publisher to register
        self.place_pose = None
        self.status = None

    def pose_cb(self, msg):
        self.place_pose = msg.data

    def event_cb(self, msg):
        self.status = msg.data

    def execute(self, userdata):
        location = Utils.get_value_of(userdata.goal.parameters, "location")
        if location is None:
            rospy.logwarn('"location" not provided. Using default.')
            return "failed"

        self.place_pose = None
        self.status = None
        self.platform_name_pub.publish(String(data=location))

        start_time = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if rospy.Time.now() - start_time > self.timeout:
                break
            if self.place_pose is not None and self.status is not None:
                break
            rate.sleep()

        if self.place_pose is not None and self.status == "e_success":
            userdata.move_arm_to = self.place_pose  
            return "succeeded"
        else:
            return "failed"


def main():
    rospy.init_node("default_place_object_server")
    sm = smach.StateMachine(outcomes=["OVERALL_SUCCESS", "OVERALL_FAILED"], input_keys=["goal"], output_keys=["result"])
    sm.userdata.move_arm_to = None

    with sm:
        smach.StateMachine.add(
            "START_PLACE_POSE_SELECTOR",
            gbs.send_event([("/mcr_perception/place_pose_selector/event_in", "e_start")]),
            transitions={"success": "GET_POSE_TO_PLACE_OBJECT"},
        )

        smach.StateMachine.add(
            "GET_POSE_TO_PLACE_OBJECT",
            GetPoseToPlaceObject(
                "/mcr_perception/place_pose_selector/platform_name",
                "/mcr_perception/place_pose_selector/place_pose",
                "/mcr_perception/place_pose_selector/event_out",
                10.0,
            ),
            transitions={
                "succeeded": "MOVE_ARM_TO_PLACE_OBJECT",
                "failed": "MOVE_ARM_TO_DEFAULT_PLACE",
            },
        )

        smach.StateMachine.add(
            "MOVE_ARM_TO_DEFAULT_PLACE",
            DefaultSafePose(),
            transitions={
                "succeeded": "MOVE_ARM_TO_PLACE_OBJECT",
                "failed": "MOVE_ARM_TO_DEFAULT_PLACE",
            },
        )
        smach.StateMachine.add(
            "MOVE_ARM_TO_PLACE_OBJECT",
            gms.move_arm(),
            transitions={"succeeded": "STOP_PLACE_POSE_SELECTOR", "failed": "STOP_PLACE_POSE_SELECTOR"},
        )

        smach.StateMachine.add(
            "STOP_PLACE_POSE_SELECTOR",
            gbs.send_event([("/mcr_perception/place_pose_selector/event_in", "e_stop")]),
            transitions={"success": "OPEN_GRIPPER"},
        )

        smach.StateMachine.add(
            "OPEN_GRIPPER",
            gms.control_gripper('open'),
            transitions={"succeeded": "OVERALL_SUCCESS", "timeout": "OVERALL_SUCCESS"},
        )

    asw = ActionServerWrapper(
        server_name="default_place_object_server",
        action_spec=GenericExecuteAction,
        wrapped_container=sm,
        succeeded_outcomes=["OVERALL_SUCCESS"],
        aborted_outcomes=["OVERALL_FAILED"],
        goal_key="goal",
        result_key="result",
    )
    asw.run_server()
    rospy.spin()

if __name__ == "__main__":
    main()