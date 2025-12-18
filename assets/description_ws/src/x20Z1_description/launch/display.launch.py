import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node


def generate_launch_description():
    package_name = "x20Z1_description"

    pkg_path = get_package_share_directory(package_name)
    xacro_file = os.path.join(pkg_path, "urdf", "x20Z1.urdf")

    robot_description_config = Command(["xacro ", xacro_file])

    params = {"robot_description": robot_description_config}

    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[params],
    )

    # node_joint_state_publisher_gui = Node(
    #     package="joint_state_publisher_gui",
    #     executable="joint_state_publisher_gui",
    #     name="joint_state_publisher_gui",
    # )

    node_rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        # arguments=['-d', os.path.join(pkg_path, 'rviz', 'config.rviz')]
    )

    node_visualizer = Node(
        package=package_name, executable="visualizer.py", name="visualizer", output="screen"
    )

    return LaunchDescription(
        [
            node_robot_state_publisher,
            # node_joint_state_publisher_gui,
            node_rviz,
            node_visualizer,
        ]
    )
