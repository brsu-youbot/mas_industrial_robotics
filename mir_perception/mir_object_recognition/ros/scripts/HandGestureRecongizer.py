#!/usr/bin/env python3
"""
HandGestureRecongizer.py

Description:
    This node is used to detect hand gestures using mediapipe model.

Author:
    Amirhossein Soltani

Created:
    2025-06-27

"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image


class GestureRecognition():
    def __init__(self, debug_mode=True):
        
        self.cvbridge = CvBridge()
        self.debug = debug_mode
        self.pub_debug = rospy.Publisher("/mir_perception/gesture_recognition/output/gesture_recognition_debug", Image, queue_size=1)

        rospy.Subscriber("/mir_perception/gesture_recognition/event_in", String, self.event_in_cb)
        self.event_out = rospy.Publisher("/mir_perception/gesture_recognition/event_out", String, queue_size=1)

        base_options = python.BaseOptions(model_asset_path='../model/gesture_recognizer.task')
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
            
            
        self.event = None
        self.gesture_sent = False
        self.frame_counter = 0
     
    # Function to add text on the frame
    @staticmethod
    def add_text(frame, text, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale=0.9, font_color=(105, 225, 0), thickness=2):
        cv2.putText(frame, text, position, font, font_scale, font_color, thickness)
    
    def callback(self, img_msg):
        try:
            self.run(img_msg)
        except Exception as e:
            rospy.loginfo("[hand_gesture_node] killing callback")

    def event_in_cb(self, msg):
        """ 
        Starts a planned motion based on the specified arm position.
        """
        self.event = msg.data
        if self.event.startswith("e_start") and not self.gesture_sent:

            self.sub_img = rospy.Subscriber("/tower_cam3d_front/rgb/image_raw", Image, self.callback)

        if self.gesture_sent:
            self.sub_img.sub.unregister()
            rospy.loginfo("[gesture_recognition_node] Unregistered image subscribers")

    def mediapipe_detect(self, cv_img):
        """ 
        Detects hand gestures using mediapipe model
        """
        imgRGB = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        mp_RGB_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        recognition_result = self.recognizer.recognize(mp_RGB_frame)
        if recognition_result.gestures:
            print(recognition_result.gestures)
            top_gesture = recognition_result.gestures[0][0].category_name
        else:
            top_gesture = None
        
        return top_gesture
    
    def run(self, image):
        """
        image: image data of current frame with Image data type
        """
        if image:
            try:
                cv_img = self.cvbridge.imgmsg_to_cv2(image, "bgr8")
                top_gesture = self.mediapipe_detect(cv_img)

                if top_gesture.upper() == "THUMB-UP":
                    self.frame_counter += 1
                
                    if self.frame_counter >= 5:
                        self.gesture_sent = True
                        msg_str = f'e_done'
                        self.event_out.publish(String(data=msg_str))
                        # print("gesture e_done sent")
                else:
                    self.frame_counter = 0

                if self.debug:
                    # Add the detected gesture to the frame
                    self.add_text(cv_img, text=f"Detected gesture: {top_gesture}", org=(450, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                    self.publish_debug_img(cv_img)

            except CvBridgeError as e:
                rospy.logerr(e)
                return
        else:
            rospy.logwarn("No image received") 

    def publish_debug_img(self, debug_img):
        debug_img = np.array(debug_img, dtype=np.uint8)
        debug_img = self.cvbridge.cv2_to_imgmsg(debug_img, "bgr8")
        self.pub_debug.publish(debug_img)


