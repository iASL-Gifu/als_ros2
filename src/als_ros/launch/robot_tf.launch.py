from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def launch_setup(context, *args, **kwargs):
    config_path = LaunchConfiguration('calibration_config').perform(context)

    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)["/**"]["ros__parameters"]

    args = [
      str(params['base_link_to_laser']['x']),
      str(params['base_link_to_laser']['y']),
      str(params['base_link_to_laser']['z']),
      str(params['base_link_to_laser']['roll']),
      str(params['base_link_to_laser']['pitch']),
      str(params['base_link_to_laser']['yaw']),
      params['base_link_to_laser']['parent_frame'],
      params['base_link_to_laser']['child_frame'],
    ]

    static_tf_node_bl = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_baselink_to_laser",
        arguments=args,
    )

    return [static_tf_node_bl]


def generate_launch_description():
    calibration_config = os.path.join(
        get_package_share_directory('als_ros2'),
        'config',
        'sensors_calibration.yaml'
    )

    calibration_la = DeclareLaunchArgument(
        'calibration_config',
        default_value=calibration_config,
        description='Path to the sensors calibration config file'
    )

    return LaunchDescription([
        calibration_la,
        OpaqueFunction(function=launch_setup)
    ])