/*
 * Copyright 2019 Bonn-Rhein-Sieg University
 *
 * Author: Mohammad Wasil
 * ROS2 contributors: Hamsa Datta Perur, Shubham Shinde, Vamsi Kalagaturu, Vivek
 * Mannava.
 *
 */

#ifndef MIR_OBJECT_RECOGNITION_MULTIMODAL_OBJECT_RECOGNITION_ROS_H
#define MIR_OBJECT_RECOGNITION_MULTIMODAL_OBJECT_RECOGNITION_ROS_H

#include <Eigen/Dense>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "rclcpp/logger.hpp"
#include "rclcpp/publisher.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp/utilities.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"
#include "rcutils/logging_macros.h"

#include "rclcpp_action/rclcpp_action.hpp"

#include "rmw/qos_profiles.h"

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include "std_msgs/msg/string.hpp"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64.hpp>

#include "mir_interfaces/msg/bounding_box_list.hpp"
#include "mir_interfaces/msg/image_list.hpp"
#include "mir_interfaces/msg/object_list.hpp"

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

#include "mir_object_segmentation/scene_segmentation_ros.hpp"
#include "mir_perception_utils/bounding_box.hpp"
#include "mir_perception_utils/bounding_box_visualizer.hpp"
#include "mir_perception_utils/clustered_point_cloud_visualizer.hpp"
#include "mir_perception_utils/label_visualizer.hpp"
#include "mir_perception_utils/object_utils_ros.hpp"
#include "mir_perception_utils/pointcloud_utils_ros.hpp"
#include "multimodal_object_recognition_utils.hpp"

#include "mir_interfaces/action/object_detection.hpp"

#include "yolo_inference.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>

using std::placeholders::_1;
using std::placeholders::_2;

using namespace std::chrono_literals;

namespace mpu = mir_perception_utils;
using mpu::visualization::BoundingBoxVisualizer;
using mpu::visualization::ClusteredPointCloudVisualizer;
using mpu::visualization::Color;
using mpu::visualization::LabelVisualizer;

struct Object
{
  std::string name;
  std::string shape;
  std::string color;
};
typedef std::vector<Object> ObjectInfo;

namespace perception_namespace {
class MultiModalObjectRecognitionROS : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit MultiModalObjectRecognitionROS(const rclcpp::NodeOptions& options);

  /// Transition callback for state configuring
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State&);

  /// Transition callback for state activating
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State& state);

  /// Transition callback for state deactivating
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State& state);

  /// Transition callback for state cleaningup
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State&);

  /// Transition callback for state shutting down
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_shutdown(const rclcpp_lifecycle::State& state);

private:
  rclcpp_action::Server<mir_interfaces::action::ObjectDetection>::SharedPtr
    action_server_;
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID& uuid,
    std::shared_ptr<const mir_interfaces::action::ObjectDetection::Goal> goal);

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<
      rclcpp_action::ServerGoalHandle<mir_interfaces::action::ObjectDetection>>
      goal_handle);

  void handle_accepted(const std::shared_ptr<rclcpp_action::ServerGoalHandle<
                         mir_interfaces::action::ObjectDetection>> goal_handle);

  void execute_goal(const std::shared_ptr<rclcpp_action::ServerGoalHandle<
                      mir_interfaces::action::ObjectDetection>> goal_handle);

  std::shared_ptr<
    rclcpp_action::ServerGoalHandle<mir_interfaces::action::ObjectDetection>>
    current_goal_handle_ = nullptr;

  OnSetParametersCallbackHandle::SharedPtr callback_handle_;

  message_filters::Subscriber<sensor_msgs::msg::Image,
                              rclcpp_lifecycle::LifecycleNode>
    image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2,
                              rclcpp_lifecycle::LifecycleNode>
    cloud_sub_;

  typedef message_filters::sync_policies::
    ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud2>
      msgSyncPolicy;
  typedef message_filters::Synchronizer<msgSyncPolicy> Sync;
  std::shared_ptr<Sync> msg_sync_;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{ nullptr };
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float64>>
    pub_workspace_height_;

  // publisher debug
  std::shared_ptr<
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>>
    pub_debug_cloud_plane_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>>
    pub_debug_rgb_image_;

  // Publisher for cloud recognizer
  std::shared_ptr<
    rclcpp_lifecycle::LifecyclePublisher<mir_interfaces::msg::ObjectList>>
    pub_cloud_to_recognizer_;

  // Callback group for subscribers
  rclcpp::CallbackGroup::SharedPtr recognized_callback_group_;

  // Subscription options
  rclcpp::SubscriptionOptions recognized_sub_options;

  // Subscriber for cloud recognizer
  std::shared_ptr<rclcpp::Subscription<mir_interfaces::msg::ObjectList>>
    sub_recognized_cloud_list_;

  // Publisher object lsit
  std::shared_ptr<
    rclcpp_lifecycle::LifecyclePublisher<mir_interfaces::msg::ObjectList>>
    pub_object_list_;

  // Publisher pose array (debug_mode only)
  std::shared_ptr<
    rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::PoseArray>>
    pub_pc_object_pose_array_;
  std::shared_ptr<
    rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::PoseArray>>
    pub_rgb_object_pose_array_;

  // --------------------------- function declarations
  // -----------------------------------

  rcl_interfaces::msg::SetParametersResult parametersCallback(
    const std::vector<rclcpp::Parameter>& parameters);

  /** @brief method to declare all parameters of this node */
  void declare_all_parameters();

  /** @brief method to load all parameters into node variables */
  void get_all_parameters();

  /**
   * \brief Callback for the synchronized image and point cloud messages
   * \param[in] image_msg
   * \param[in] cloud_msg
   */
  void synchronizeCallback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                           const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg,
                           const std::string workstation);

  // Recognize Clouds callback
  void recognizedCloudCallback(const mir_interfaces::msg::ObjectList& msg);

  /** \brief Transform pointcloud to the given target frame id ("base_link" by
   * default) \param[in] PointCloud2 input
   */
  void preprocessPointCloud(
   const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg);

  /** \brief Recognize 2D and 3D objects, estimate their pose, filter them, and
   * publish the object_list */
  virtual void recognizeCloudAndImage(const std::string workstation);

  /** \brief Adjust object pose, make it flat, adjust container, axis and bolt
   *poses. \param[in] Object_list
   *
   **/
  void adjustObjectPose(mir_interfaces::msg::ObjectList& object_list);

  /** \brief Publish object_list to object_list merger
   * \param[in] Object_list to publish
   **/
  void publishObjectList(mir_interfaces::msg::ObjectList& object_list);

  /** \brief Publish debug info such as bbox, poses, labels for both 2D and 3D
   *objects. \param[in] combined object list \param[in] 3D pointcloud cluster
   *from 3D object segmentation \param[in] 3D pointcloud cluster from 2D
   *bounding box proposal
   **/
  void publishDebug(mir_interfaces::msg::ObjectList& combined_object_list,
                    std::vector<PointCloudBSPtr>& clusters_3d,
                    std::vector<PointCloudBSPtr>& clusters_2d);

  /** \brief Load qualitative object info
   * \param[in] Path to the yaml object file
   * */
  void loadObjectInfo(const std::string& filename);

protected:
  typedef std::shared_ptr<SceneSegmentationROS> SceneSegmentationROSSPtr;
  SceneSegmentationROSSPtr scene_segmentation_ros_;
  typedef std::shared_ptr<MultimodalObjectRecognitionUtils>
    MultimodalObjectRecognitionUtilsSPtr;
  MultimodalObjectRecognitionUtilsSPtr mm_object_recognition_utils_;
  typedef std::shared_ptr<YoloInference> YoloInferenceSPtr;
  YoloInferenceSPtr yolo_inference_;

  // visualization
  std::shared_ptr<BoundingBoxVisualizer> bounding_box_visualizer_pc_;
  std::shared_ptr<ClusteredPointCloudVisualizer> cluster_visualizer_rgb_;
  std::shared_ptr<ClusteredPointCloudVisualizer> cluster_visualizer_pc_;
  std::shared_ptr<LabelVisualizer> label_visualizer_rgb_;
  std::shared_ptr<LabelVisualizer> label_visualizer_pc_;

  // parameters
  bool debug_mode_;
  std::string pointcloud_source_frame_id_;
  std::string target_frame_id_;
  std::set<std::string> round_objects_;

  ObjectInfo object_info_;
  std::string objects_info_path_;

  // yolo inference file paths
  std::string yolo_classes_info_path_;
  std::string yolo_weights_path_;

  // Used to store pointcloud and image received from callback
  sensor_msgs::msg::PointCloud2::ConstSharedPtr pointcloud_msg_;
  sensor_msgs::msg::Image::ConstSharedPtr image_msg_;
  PointCloudBSPtr cloud_;

  // Dynamic parameter
  double voxel_leaf_size_;
  std::string voxel_filter_field_name_;
  double voxel_filter_limit_min_;
  double voxel_filter_limit_max_;
  bool enable_passthrough_filter_;
  std::string passthrough_filter_field_name_;
  double passthrough_filter_limit_min_;
  double passthrough_filter_limit_max_;
  bool enable_cropbox_filter_;
  double cropbox_filter_x_limit_min_;
  double cropbox_filter_x_limit_max_;
  double cropbox_filter_y_limit_min_;
  double cropbox_filter_y_limit_max_;
  double cropbox_filter_z_limit_min_;
  double cropbox_filter_z_limit_max_;
  double normal_radius_search_;
  bool use_omp_;
  int num_cores_;
  int sac_max_iterations_;
  double sac_distance_threshold_;
  bool sac_optimize_coefficients_;
  double sac_x_axis_;
  double sac_y_axis_;
  double sac_z_axis_;
  double sac_eps_angle_;
  double sac_normal_distance_weight_;
  double prism_min_height_;
  double prism_max_height_;
  double outlier_radius_search_;
  int outlier_min_neighbors_;
  double cluster_tolerance_;
  int cluster_min_size_;
  int cluster_max_size_;
  double cluster_min_height_;
  double cluster_max_height_;
  double cluster_max_length_;
  double cluster_min_distance_to_polygon_;

  // cluster
  bool center_cluster_;
  bool pad_cluster_;
  int padded_cluster_size_;

  // logdir for saving debug image
  std::string logdir_;

  // rgb_object_id used to differentiate 2D and 3D objects
  int rgb_object_id_;

  double octree_resolution_;
  double object_height_above_workspace_;
  double container_height_;

  // Flags for object recognition
  bool received_recognized_cloud_list_flag_;

  // Recognized image list
  mir_interfaces::msg::ObjectList recognized_cloud_list_;
  mir_interfaces::msg::ObjectList recognized_image_list_;

  // Enable recognizer
  bool enable_rgb_recognizer_;
  bool enable_pc_recognizer_;

  int rgb_roi_adjustment_;
  int rgb_bbox_min_diag_;
  int rgb_bbox_max_diag_;
  double rgb_cluster_filter_limit_min_;
  double rgb_cluster_filter_limit_max_;
  bool rgb_cluster_remove_outliers_;
  bool enable_roi_;
  double roi_base_link_to_laser_distance_;
  double roi_max_object_pose_x_to_base_link_;
  double roi_min_bbox_z_;

  // QoS profiles
  rclcpp::SensorDataQoS qos_sensor;
  rclcpp::ParametersQoS qos_parameters;
  rclcpp::SystemDefaultsQoS qos_default;

  /** \brief Add cloud accumulation, segment accumulated pointcloud, find the
   *plane, clusters table top objects, find object heights. \param[out] 3D
   *object list with unknown label \param[out] Tabletop pointcloud clusters
   **/
  void segmentPointCloud(mir_interfaces::msg::ObjectList& object_list,
                         std::vector<PointCloudBSPtr>& clusters,
                         std::vector<mpu::object::BoundingBox>& boxes);
};

} // namespace perception_namespace ends
#endif // MIR_OBJECT_RECOGNITION_MULTIMODAL_OBJECT_RECOGNITION_ROS_H
