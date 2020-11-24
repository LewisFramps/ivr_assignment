#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import math

class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)

    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    self.im2 = None
    self.initial_time = rospy.get_time()

  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)

    im1=cv2.imshow('window1', self.cv_image1)

    cv2.waitKey(1)

    self.redMask  = self.get_red_mask(self.cv_image1)
    self.blueMask = self.get_blue_maks(self.cv_image1)
    self.yellowMask = self.get_yellow_mask(self.cv_image1)
    self.greenMask = self.get_green_mask(self.cv_image1)

    foundJoints = [self.colourInImage(self.yellowMask), self.colourInImage(self.blueMask),
                   self.colourInImage(self.greenMask), self.colourInImage(self.redMask)]

    self.redCentre = None
    self.blueCentre = None
    self.yellowCentre = None
    self.greenCentre = None

    if foundJoints[0]: self.yellowCentre = self.get_pixel_centre(self.yellowMask)
    if foundJoints[1]: self.blueCentre = self.get_pixel_centre(self.blueMask)
    if foundJoints[2]: self.greenCentre = self.get_pixel_centre(self.greenMask)
    if foundJoints[3]: self.redCentre = self.get_pixel_centre(self.redMask)

    test = np.zeros((self.cv_image1.shape[0], self.cv_image1.shape[1], 3), np.uint8)

    test[self.redCentre[1]][self.redCentre[0]] = (255,255,255)
    test[self.yellowCentre[1]][self.yellowCentre[0]] = (255,255,255)
    test[self.blueCentre[1]][self.blueCentre[0]] = (255,255,255)
    test[self.greenCentre[1]][self.greenCentre[0]] = (255,255,255)
    kernal = np.ones((5,5), np.uint8)
    x = cv2.imshow("pls", cv2.dilate(test, kernal, iterations=3))

    # Update joint angles
    joints = self.jointMovement()
    self.joint2=Float64()
    self.joint3=Float64()
    self.joint4=Float64()

    self.joint2 = joints[0]
    self.joint3 = joints[1]
    self.joint4 = joints[2]


    im1=cv2.imshow('window1', self.cv_image1)
    ay = cv2.imshow('xxx', self.image2)
    cv2.waitKey(1)
    # Publish the results
    try:
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      print(joints[0])
      self.robot_joint2_pub.publish(self.joint2);
      self.robot_joint3_pub.publish(self.joint3);
      self.robot_joint4_pub.publish(self.joint4);
    except CvBridgeError as e:
      print(e)

  def callback2(self, data):
    try:
      self.image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

  # check an image to see if we can see any colour at all
  def colourInImage(self, image):
    return cv2.countNonZero(image) != 0

  def get_red_mask(self, image):
      # Only grab red pixels
      mask = cv2.inRange(image, (0, 0, 100), (0, 0, 225))

      kernal = np.ones((5,5), np.uint8)
      return cv2.dilate(mask, kernal, iterations=3)

  def get_blue_maks(self, image):
    # Only grab blue pixels
    mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
    kernal = np.ones((5,5), np.uint8)
    return cv2.dilate(mask, kernal, iterations=3)

  def get_yellow_mask(self, image):
    # Only grab yellow pixels
    hsvIm = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvIm, (20, 100, 100), (30, 255, 255))
    kernal = np.ones((5,5), np.uint8)
    return cv2.dilate(mask, kernal, iterations=3)

  def get_green_mask(self, image):
    # Only grab green pixels
    mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
    kernal = np.ones((5,5), np.uint8)
    return cv2.dilate(mask, kernal, iterations=3)

  def get_pixel_centre(self, mask):
    M = cv2.moments(mask)
    if M['m00'] == 0: return np.array([0,0])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])

  def jointMovement(self):
    pi = math.pi
    t = rospy.get_time() - self.initial_time
    j2 = (pi/3.0) * math.sin((pi/15.0) * t)
    j3 = (pi/3.0) * math.sin((pi/18.0) * t)
    j4 = (pi/3.0) * math.sin((pi/20.0) * t)
    return [j2, j3, j4]

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


