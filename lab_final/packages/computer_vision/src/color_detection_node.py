#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from duckietown.dtros import DTROS, NodeType

class ColorDetectionNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(ColorDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )
        
        # Get vehicle name from environment variable
        self.vehicle_name = os.environ['VEHICLE_NAME']
        
        # Camera info and image topics
        self.camera_info_topic = f"/{self.vehicle_name}/camera_node/camera_info"
        self.distorted_img_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        
        # Bridge for converting between ROS and OpenCV images
        self.bridge = CvBridge()
        
        # Camera parameters (will be updated from camera_info)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.got_camera_info = False
        
        # Image processing parameters
        self.resize_width = 320  # Set your desired width
        self.resize_height = 240  # Set your desired height
        self.blur_kernel_size = (5, 5)  # Gaussian blur kernel size
        self.blur_sigma = 0  # 0 means auto-calculate based on kernel size
        
        # HSV color thresholds for each line color
        # These are initial values - you should adjust them based on your captured images
        self.hsv_thresholds = {
            'blue': {
                'lower': np.array([100, 80, 80]),
                'upper': np.array([130, 255, 255])
            },
            'red': {  # Red wraps around the hue space
                'lower1': np.array([0, 100, 100]),
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([160, 100, 100]),
                'upper2': np.array([180, 255, 255])
            },
            'green': {
                'lower': np.array([40, 80, 80]),
                'upper': np.array([80, 255, 255])
            }
        }
        
        # Save image flag for capturing reference images
        self.save_images = False
        self.save_counter = 0
        
        # Create publishers for processed images
        self.undistorted_pub = rospy.Publisher(
            f"/{self.vehicle_name}/camera_node/image/undistorted",
            Image,
            queue_size=1
        )
        
        self.processed_pub = rospy.Publisher(
            f"/{self.vehicle_name}/camera_node/image/processed",
            Image,
            queue_size=1
        )
        
        self.detected_pub = rospy.Publisher(
            f"/{self.vehicle_name}/camera_node/image/detected",
            Image,
            queue_size=1
        )
        
        # Subscribe to camera info topic
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self.camera_info_callback
        )
        
        # Subscribe to compressed image topic
        self.image_sub = rospy.Subscriber(
            self.distorted_img_topic,
            CompressedImage,
            self.image_callback
        )
        
        # Set processing rate to 5 Hz as suggested in the exercise
        self.rate = rospy.Rate(5)
        
        self.log("Color detection node initialized")
    
    def camera_info_callback(self, msg):
        """Extract camera intrinsic parameters from camera_info topic"""
        if not self.got_camera_info:
            # Extract the camera matrix (K) from the message
            self.camera_matrix = np.array(msg.K).reshape((3, 3))
            
            # Extract distortion coefficients (D) from the message
            self.dist_coeffs = np.array(msg.D)
            
            self.got_camera_info = True
            self.log(f"Received camera intrinsic parameters")
    
    def resize_image(self, image):
        """Resize image to the specified dimensions"""
        return cv2.resize(image, (self.resize_width, self.resize_height))
    
    def apply_blur(self, image):
        """Apply Gaussian blur to the image"""
        return cv2.GaussianBlur(image, self.blur_kernel_size, self.blur_sigma)
    
    def detect_color(self, hsv_image, color_name):
        """Detect specified color in HSV image and return binary mask"""
        if color_name == 'red':
            # Red color wraps around the HSV hue space
            mask1 = cv2.inRange(hsv_image, self.hsv_thresholds['red']['lower1'], 
                                self.hsv_thresholds['red']['upper1'])
            mask2 = cv2.inRange(hsv_image, self.hsv_thresholds['red']['lower2'], 
                                self.hsv_thresholds['red']['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_image, self.hsv_thresholds[color_name]['lower'], 
                              self.hsv_thresholds[color_name]['upper'])
            
        # Apply morphology operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def find_contours(self, mask):
        """Find contours in binary mask and return the largest one"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        
        if not filtered_contours:
            return None
            
        # Return the largest contour
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        return largest_contour
    
    def draw_detection_info(self, image, contour, color_name):
        """Draw bounding box and info about the detected lane"""
        if contour is None:
            return image
            
        # Define color for drawing (BGR format)
        draw_colors = {
            'blue': (255, 0, 0),    # Blue
            'green': (0, 255, 0),   # Green
            'red': (0, 0, 255)      # Red
        }
        
        # Get rectangle bounding the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw rectangle on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), draw_colors[color_name], 2)
        
        # Put text label
        cv2.putText(image, f"{color_name} lane", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_colors[color_name], 2)
        
        # Calculate lane dimensions
        # These are in pixels, you would need to convert to real-world units
        lane_width = w  # Width in pixels
        lane_height = h  # Height in pixels
        
        # Display dimensions
        dimension_text = f"W: {lane_width}px, H: {lane_height}px"
        cv2.putText(image, dimension_text, (x, y + h + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_colors[color_name], 2)
        
        return image, (x, y, w, h)
    
    def save_color_image(self, image, color_name):
        """Save image of a specific color lane for reference"""
        if not os.path.exists('/data/color_samples'):
            os.makedirs('/data/color_samples')
        
        filename = f"/data/color_samples/{color_name}_lane_{self.save_counter}.jpg"
        cv2.imwrite(filename, image)
        self.log(f"Saved {color_name} lane image to {filename}")
    
    def image_callback(self, msg):
        """Process incoming compressed images, detect colors, and publish results"""
        # Skip if we haven't received camera parameters yet
        if not self.got_camera_info:
            return
        
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            distorted_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Get image dimensions
            h, w = distorted_img.shape[:2]
            
            # Calculate optimal camera matrix
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 0, (w, h)
            )
            
            # Manual undistortion using cv2.undistort
            undistorted_img = cv2.undistort(
                distorted_img, 
                self.camera_matrix, 
                self.dist_coeffs, 
                None, 
                new_camera_matrix
            )
            
            # Optional: Crop the image to the ROI
            x, y, w, h = roi
            if all([x, y, w, h]):  # Make sure ROI is valid
                undistorted_img = undistorted_img[y:y+h, x:x+w]
            
            # Publish undistorted image
            undistorted_msg = self.bridge.cv2_to_imgmsg(undistorted_img, "bgr8")
            undistorted_msg.header = msg.header
            self.undistorted_pub.publish(undistorted_msg)
            
            # Apply pre-processing: resize the image
            resized_img = self.resize_image(undistorted_img)
            
            # Apply pre-processing: blur the image
            processed_img = self.apply_blur(resized_img)
            
            # Publish processed image
            processed_msg = self.bridge.cv2_to_imgmsg(processed_img, "bgr8")
            processed_msg.header = msg.header
            self.processed_pub.publish(processed_msg)
            
            # Convert to HSV for color detection
            hsv_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
            
            # Make a copy of the processed image for drawing detection results
            detection_img = processed_img.copy()
            
            # Dictionary to store lane dimensions
            lane_dimensions = {}
            
            # Detect each color and draw bounding boxes
            for color_name in ['blue', 'green', 'red']:
                # Detect color
                color_mask = self.detect_color(hsv_img, color_name)
                
                # Find contours
                largest_contour = self.find_contours(color_mask)
                
                # Draw detection information if contour was found
                if largest_contour is not None:
                    detection_img, dimensions = self.draw_detection_info(
                        detection_img, largest_contour, color_name
                    )
                    lane_dimensions[color_name] = dimensions
                    
                    # Save images if flag is set
                    if self.save_images:
                        # Create a mask of just this contour
                        contour_mask = np.zeros_like(color_mask)
                        cv2.drawContours(contour_mask, [largest_contour], 0, 255, -1)
                        
                        # Extract the region from the original image
                        x, y, w, h = dimensions
                        color_sample = processed_img[y:y+h, x:x+w]
                        
                        # Save the image
                        self.save_color_image(color_sample, color_name)
            
            # If save images flag was set, increment counter and reset flag
            if self.save_images:
                self.save_counter += 1
                self.save_images = False
            
            # Log lane dimensions
            if lane_dimensions:
                self.log(f"Detected lane dimensions: {lane_dimensions}")
            
            # Publish detection results
            detection_msg = self.bridge.cv2_to_imgmsg(detection_img, "bgr8")
            detection_msg.header = msg.header
            self.detected_pub.publish(detection_msg)
            
        except Exception as e:
            self.logerr(f"Error processing image: {e}")
    
    def save_reference_images(self):
        """Set flag to save reference images of detected lanes"""
        self.save_images = True
        self.log("Will save images of the next detected lines")
    
    def run(self):
        """Main loop to keep the node running at the specified rate"""
        self.log("Starting color detection node")
        while not rospy.is_shutdown():
            self.rate.sleep()

if __name__ == '__main__':
    # Initialize the node
    color_node = ColorDetectionNode(node_name='color_detection_node')
    
    # Set up image saving if needed
    # Uncomment to enable saving a set of images
    # rospy.Timer(rospy.Duration(5), lambda _: color_node.save_reference_images())
    
    # Run the node
    color_node.run()
