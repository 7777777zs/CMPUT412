#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._annotated_topic = f"/{self._vehicle_name}/camera_node/annotated_image/compressed"
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        # create window
        self._window = "camera-reader"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
                # construct publisher
        self.pub = rospy.Publisher(self._annotated_topic, CompressedImage, queue_size=1)


    def callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        height, width = image.shape[:2]
        rospy.loginfo(f"image size: {width}x{height}")

        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image  # Already grayscale

        annotated_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        # add text annotation
        text = f"Duck {self._vehicle_name} says, 'Cheese! Capturing {width}x{height} - quack-tastic!'"
        cv2.putText(annotated_image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # display frame
        cv2.imshow(self._window, annotated_image)
        cv2.waitKey(1)

        # convert back to compressed image and publish
        annotated_msg = self._bridge.cv2_to_compressed_imgmsg(annotated_image)
        self.pub.publish(annotated_msg)

if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()
