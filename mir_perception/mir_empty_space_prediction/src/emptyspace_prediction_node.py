import rospy
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import yaml
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from threading import Lock
import os
import rospkg

class EmptySpaceDetector:
    def __init__(self):
        rospy.init_node("empty_space_detector")

        self.event_sub = rospy.Subscriber("/empty_space_detector/event_in", String, self.event_callback)
        self.event_pub = rospy.Publisher("/empty_space_detector/event_out", String, queue_size=10)

        self.process_image = False
        self.process_pointcloud = False  # Corrected typo

        self.lock = Lock()

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('mir_empty_space_prediction')

        self.model_path = os.path.join(package_path, "model/YOLOv8s.pt")
        
        # self.model_path = "/home/anudeep/b_it_bots/empty_space_models/YOLOv8s.pt"

        # Use the params.yaml file to define the ROI
        # roi_params_path = '/home/anudeep/b_it_bots/src/emptyspace_prediction/config/params.yaml'

        roi_params_path = os.path.join(package_path, "config/params.yaml")

        with open(roi_params_path, 'r') as file:
            roi_params = yaml.safe_load(file)
        self.roi = (roi_params['x_min'], roi_params['x_max'], roi_params['y_min'], roi_params['y_max'])

        self.model = YOLO(self.model_path)
        
        self.bridge = CvBridge()
        self.latest_image = None
        self.center = None
        self.best_box = None
        
        rospy.Subscriber("/tower_cam3d_front/color/image_raw", Image, self.image_callback)
        rospy.Subscriber("/tower_cam3d_front/depth/color/points", PointCloud2, self.pointcloud_callback)
        self.marker_pub = rospy.Publisher("/projected_point_marker", Marker, queue_size=10)
        self.debug_image_pub = rospy.Publisher("/debug_empty_space", Image, queue_size=10)
        self.empty_space_pub = rospy.Publisher("/empty_space_pose", PoseStamped, queue_size=10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def event_callback(self, msg):
        with self.lock:
            if msg.data == "e_empty":
                self.process_image = True
                self.process_pointcloud = False
            elif msg.data == "e_cloud":
                self.process_pointcloud = True
                self.process_image = False
            elif msg.data == "e_stop":
                self.process_image = False
                self.process_pointcloud = False
    
    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        with self.lock:
            if self.process_image:
                self.predict_empty_space()
    
    def pointcloud_callback(self, msg):
        with self.lock:
            if self.process_pointcloud and self.center is not None:
                self.process_pointcloud_data(msg)
    
    def predict_empty_space(self):
        if self.latest_image is None:
            return
        
        results = self.model(self.latest_image)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        roi_boxes = [box for box in boxes if self.is_box_in_roi(box)]
        if not roi_boxes:
            self.center = None
            self.best_box = None
            return
        
        self.best_box = max(roi_boxes, key=lambda box: self.box_area(box))
        self.center = self.box_center(self.best_box)
        rospy.loginfo(f"Best empty space center: {self.center}")
        
        self.publish_debug_image()
        self.event_pub.publish(String("e_empty_space_detected"))
    
    def is_box_in_roi(self, box):
        return (box[0] >= self.roi[0] and box[2] <= self.roi[1] and
                box[1] >= self.roi[2] and box[3] <= self.roi[3])
    
    def box_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])
    
    def box_center(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def publish_debug_image(self):
        if self.latest_image is None or self.best_box is None:
            return
        
        debug_image = self.latest_image.copy()
        
        # Draw ROI
        cv2.rectangle(debug_image, (self.roi[0], self.roi[2]), (self.roi[1], self.roi[3]), (0, 0, 255), 2)
        
        # Draw best box
        cv2.rectangle(debug_image, 
                      (int(self.best_box[0]), int(self.best_box[1])), 
                      (int(self.best_box[2]), int(self.best_box[3])), 
                      (0, 255, 0), 2)
        
        # Draw center point
        cv2.circle(debug_image, (int(self.center[0]), int(self.center[1])), 5, (255, 0, 0), -1)
        
        # Publish debug image
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
        self.debug_image_pub.publish(debug_msg)
    
    def project_pixel_to_pointcloud(self, cloud_msg, pixel_x, pixel_y):
        width = cloud_msg.width
        height = cloud_msg.height

        if width <= 1 or height <= 1:
            rospy.logerr("PointCloud is unordered. Cannot directly map pixel coordinates.")
            return None

        points_list = list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=False))
        points_array = np.array(points_list).reshape(height, width, 3)
        reshaped_points = np.zeros((480, 640, 3))

        for i in range(3):
            reshaped_points[:,:,i] = cv2.resize(points_array[:,:,i], (640, 480), interpolation=cv2.INTER_LINEAR)

        return reshaped_points[int(pixel_y), int(pixel_x)]

    def process_pointcloud_data(self, cloud_msg):
        if self.center is None:
            return

        pixel_x, pixel_y = self.center
        projected_point = self.project_pixel_to_pointcloud(cloud_msg, pixel_x, pixel_y)

        if projected_point is not None and not np.isnan(projected_point).any():
            x, y, z = projected_point
            # rospy.loginfo(f"Projected 3D Point: x={x}, y={y}, z={z}")

            marker = Marker()
            marker.header.frame_id = cloud_msg.header.frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "projected_point"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.05
            marker.color.r = 1.0
            marker.color.a = 1.0

            self.marker_pub.publish(marker)

            pose_stamp = PoseStamped()
            pose_stamp.header.frame_id = cloud_msg.header.frame_id
            pose_stamp.header.stamp = rospy.Time.now()
            pose_stamp.pose.position.x = x
            pose_stamp.pose.position.y = y
            pose_stamp.pose.position.z = z
            pose_stamp.pose.orientation.w = 1.0


            try:
                transformed_pose = self.tf_buffer.transform(pose_stamp, "base_link", rospy.Duration(1.0))
                self.empty_space_pub.publish(transformed_pose)
                rospy.loginfo("Published empty space pose in base frame : " + str(transformed_pose.pose.position))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Transform error: {e}")

            self.event_pub.publish(String("e_pointcloud_processed"))
        else:
            rospy.logwarn("No valid point found for the given pixel coordinates.")

if __name__ == "__main__":
    detector = EmptySpaceDetector()
    rospy.spin()
