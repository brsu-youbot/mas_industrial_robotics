"""
Copyright 2022 Bonn-Rhein-Sieg University

Author: Vamsi Kalagaturu, Vivek Mannava

"""
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, EmitEvent, LogInfo
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.events.lifecycle import ChangeState
from launch_ros.events.lifecycle import matches_node_name
from ament_index_python.packages import (
    get_package_share_directory,
    get_package_prefix,
    get_resource,
)
from launch.event_handlers.on_shutdown import OnShutdown


import lifecycle_msgs.msg
import os


def generate_launch_description():
    ld = LaunchDescription()
    node_name = "world_model_node"

    arena_info = os.path.join(
        get_package_share_directory("mir_world_model"),
        "config",
        "arena.yaml",
    )

    container = ComposableNodeContainer(
        name="world_model_node_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="mir_world_model",
                plugin="WorldModelNode",
                name=node_name,
                namespace="",
                parameters=[arena_info]
            )
        ],
        # parameters=[],
        output="screen",
        # prefix=['xterm -e gdb -ex run --args'],
    )

    # transition mmor node to shutdown before SIGINT
    shutdown_event = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_node_name(
                            node_name=node_name
                        ),
                        transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVE_SHUTDOWN,
                    )
                ),
                LogInfo(msg="[mmor_launch] mmor node is exiting."),
            ],
        )
    )

    ld.add_action(shutdown_event)
    ld.add_action(container)
    return ld
