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

    # colour_closest in format [Closest colour in YZ, Closest colour in XZ]
    # Used in case we can't see target colour, use the closest objects coordinates instead
    # Initialise to centre points just to clear any unforeseen errors.
    self.redClosest = np.array([[0,0],[0,0]])
    self.greenClosest = np.array([[0,0],[0,0]])
    self.blueClosest = np.array([[0,0],[0,0]])
    self.YellowClosest = np.array([[0,0],[0,0]])
    self.sphereClosest = np.array([[0,0],[0,0]])
    self.cubeClosest = np.array([[0,0],[0,0]])

    self.joint2=Float64()
    self.joint3=Float64()
    self.joint4=Float64()

    self.pixToMet = None
    self.looped = False

    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

    self.cv_image2 = None
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



    # colourMasks in format [redMask(YZ), redMask(XZ)]
    self.redMask   = [self.get_red_mask(self.cv_image1), self.get_red_mask(self.cv_image2)]
    self.blueMask   = [self.get_blue_mask(self.cv_image1), self.get_blue_mask(self.cv_image2)]
    self.yellowMask = [self.get_yellow_mask(self.cv_image1), self.get_yellow_mask(self.cv_image2)]
    self.greenMask  = [self.get_green_mask(self.cv_image1), self.get_green_mask(self.cv_image2)]
    self.orangeMask = [self.get_orange_mask(self.cv_image1), self.get_orange_mask(self.cv_image2)]

    # get the visibility of each colour. colVis = [is seen in YZ, is seen in XZ]
    self.redVis = self.canSee(self.redMask)
    self.blueVis = self.canSee(self.blueMask)
    self.yellowVis = self.canSee(self.yellowMask)
    self.greenVis = self.canSee(self.greenMask)
    self.orangeVis = self.canSee(self.orangeMask)

    self.beans = [0,0,0]

    self.redPixCentres = self.get_pixel_centre(self.redMask, self.redVis, self.redClosest)
    self.bluePixCentres = self.get_pixel_centre(self.blueMask, self.blueVis, self.blueClosest)
    self.yellowPixCentres = self.get_pixel_centre(self.yellowMask, self.yellowVis, self.YellowClosest)
    self.greenPixCentres = self.get_pixel_centre(self.greenMask, self.greenVis, self.greenClosest)
    orangePixCentresYZ = self.orangeObjectsPixelPos(self.orangeMask[0])
    orangePixCentresXZ = self.orangeObjectsPixelPos(self.orangeMask[1])
    self.targetPixCentres = np.array([orangePixCentresYZ[0], orangePixCentresXZ[0]])
    self.cubePixCentres = np.array([orangePixCentresYZ[1], orangePixCentresXZ[1]])

    """
    print("redPixCentres: " + str(self.redPixCentres))
    print("bluePixCentres: " + str(self.bluePixCentres))
    print("yellowPixCentres: " + str(self.yellowPixCentres))
    print("greenPixCentres: " + str(self.greenPixCentres))
    print("targetPixCentres: " + str(self.targetPixCentres))
    print("cubePixCentres: " + str(self.cubePixCentres))
    print("----")
    """

    # get pixel to meter ratio only once, do this before any joints have moved.
    if not self.looped:
      self.pixToMet = self.pixel2Meter(self.yellowPixCentres[0], self.redPixCentres[0])
      self.looped = True

    # Get 2D object locations in meters on the YZ and the XZ
    self.red2D  = np.array([self.pixToMet * self.redPixCentres[0], self.pixToMet * self.redPixCentres[1]])
    self.blue2D = np.array([self.pixToMet * self.bluePixCentres[0], self.pixToMet * self.bluePixCentres[1]])
    self.yellow2D = np.array([self.pixToMet * self.yellowPixCentres[0], self.pixToMet * self.yellowPixCentres[1]])
    self.green2D = np.array([self.pixToMet * self.greenPixCentres[0], self.pixToMet * self.greenPixCentres[1]])
    self.target2D = np.array([self.pixToMet * self.targetPixCentres[0], self.pixToMet * self.targetPixCentres[1]])
    self.cube2D = np.array([self.pixToMet * self.cubePixCentres[0], self.pixToMet * self.cubePixCentres[1]])

    """
    print("red2D: " + str(self.red2D))
    print("blue2D: " + str(self.blue2D))
    print("yellow2D: " + str(self.yellow2D))
    print("green2D: " + str(self.green2D))
    print("target2D: " + str(self.target2D))
    print("cube2D: " + str(self.cube2D))
    print("---")
    """
    # get 3D coordinates in meters for each object
    self.red3D = self.get3Dim(self.red2D)
    self.blue3D = self.get3Dim(self.blue2D)
    self.yellow3D = self.get3Dim(self.yellow2D)
    self.green3D = self.get3Dim(self.green2D)
    self.target3D = self.get3Dim(self.target2D)
    self.cube3D = self.get3Dim(self.cube2D)


    print("red3D: " + str(self.red3D))
    print("blue3D: " + str(self.blue3D))
    print("yellow3D: " + str(self.yellow3D))
    print("green3D: " + str(self.green3D))
    print("target3D: " + str(self.target3D))
    print("cube3D: " + str(self.cube3D))
    print("---")


    """
    # uncomment to show image showing the centre of each point in the YZ perspective
    test = np.zeros((self.cv_image1.shape[0], self.cv_image1.shape[1], 3), np.uint8)

    test[self.redPixCentres[0][1]][self.redPixCentres[0][0]] = (255,255,255)
    test[self.bluePixCentres[0][1]][self.bluePixCentres[0][0]] = (255,255,255)
    test[self.yellowPixCentres[0][1]][self.yellowPixCentres[0][0]] = (255,255,255)
    test[self.greenPixCentres[0][1]][self.greenPixCentres[0][0]] = (255,255,255)
    test[self.targetPixCentres[0][1]][self.targetPixCentres[0][0]] = (255,255,255)
    test[self.cubePixCentres[0][1]][self.cubePixCentres[0][0]] = (255,255,255)
    kernal = np.ones((5,5), np.uint8)
    x = cv2.imshow("pls", cv2.dilate(test, kernal, iterations=3))
    """

    print("j2 calc = " + str(self.joint2))


    self.calcJointAngles([self.yellow3D, self.blue3D, self.green3D, self.red3D])
    # Update joint angles
    joints = self.jointMovement()
    self.joint2 = joints[0]
    self.joint3 = joints[1]
    self.joint4 = joints[2]
    print("Joints:   J2:" + str(np.round(10000 * joints[0]) /10000) + ",  J3:" + str(np.round(10000 * joints[1]) /10000) + ",  J4:" + str(np.round(10000 * joints[2]) /10000))

    #print(len(cont))
    imcopy = cv2.cvtColor(self.orangeMask[0].copy(), cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(imcopy, cont, -1, (0,255,75), 3)
    im1 = cv2.imshow('yzImage', self.cv_image1)
    im2 = cv2.imshow('xz,Image', self.cv_image2)
    #im2 = cv2.imshow('xzImage', self.cv_image2)
    #w1 = cv2.imshow('cimg', imcopy)
    #w2 = cv2.imshow('xz', self.orangeMask[1])
    cv2.waitKey(1)
    # Publish the results
    try:
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      self.robot_joint2_pub.publish(self.joint2);
      self.robot_joint3_pub.publish(self.joint3);
      self.robot_joint4_pub.publish(self.joint4);
    except CvBridgeError as e:
      print(e)

  def orangeObjectsPixelPos(self, image):
    targetPixelPos = np.array([-1, -1])
    cubePixelPos = np.array([-1, -1])
    targetContours = None
    cubeContours = None

    # Apply a orange mask and find contours of orange objects
    cont, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if no orange objects are found we just set the targetsPixelPos to [0,0]. This is just for testing and I don't
    # think it can actually happen while running. If I have time I'll come back to this later
    if(len(cont) == 0):
      print("this really shouldn't happen")
      return np.array([[0,0],[0,0]])

    # if objects are overlapping in view we just take the centre of both. This skew the results this is a rare
    # occurrence and if it does happen, it's usually for a tiny amount of time.
    elif(len(cont) == 1):
      targetContours = cont[0]
      cubeContours = cont[0]

    # If we find two objects, figure out which is the sphere and which is the cube and return their positions.
    # We know the circle contour should have a ton more vertices than the cube contour, so we compare those.
    elif(len(cont) == 2):
      if(len(cont[0]) > len(cont[1])):
        targetContours = cont[0]
        cubeContours = cont[1]
      else:
        targetContours = cont[1]
        cubeContours = cont[0]

    # this REALLY shouldn't happen but if it does then just return the last valid position
    elif len(cont) > 2:
      print("something's broken ! :D")
      return np.array([[0,0],[0,0]])

    # get the moments of the contours
    targM = cv2.moments(targetContours)
    cubeM = cv2.moments(cubeContours)

    # get centre position of each object
    if(targM['m00'] == 0): targetPixelPos = np.array([0,0])
    else:
      targetPixelPos[0] = int(targM['m10'] / targM['m00'])
      targetPixelPos[1] = 800 - int(targM['m01'] / targM['m00'])

    if(cubeM['m00'] == 0): cubePixelPos = np.array([0,0])
    else:
      cubePixelPos[0] = int(cubeM['m10'] / cubeM['m00'])
      cubePixelPos[1] = 800 - int(cubeM['m01'] / cubeM['m00'])

    # return target & cubes centres
    return np.array([targetPixelPos, cubePixelPos])

  def canSee(self, data):
    return np.array([self.colourInImage(data[0]), self.colourInImage(data[1])])

  def callback2(self, data):
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
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

  def get_blue_mask(self, image):
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

  def get_orange_mask(self, image):
    # Only grab orange pixels
    hsvIm = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvIm, (1, 150, 80), (18, 255, 255))
    kernal = np.ones((5,5), np.uint8)
    return cv2.dilate(mask, kernal, iterations=1)

  def get_pixel_centre(self, mask, seen, closest):
    yz = None
    xz = None

    # Find mask pixel centre in yz plane

    # if the colour doesn't show in the plane, take the colour that was closest to it before it disappeared
    if not seen[0]:
      yz = closest[0]
    else:
      M = cv2.moments(mask[0])
      if M['m00'] == 0:
        yz = np.array([0, 0])
      else:
        yz = np.array( [int(M['m10'] / M['m00']),
                        800-int(M['m01'] / M['m00']) ] )

    # Find mask pixel centre in xz plane
    # if the colour doesn't show in the plane, take the colour that was closest to it before it disappeared
    if not seen[1]:
      xz = closest[1]
    else:
      M = cv2.moments(mask[1])
      if M['m00'] == 0:
        xz = np.array([0, 0])
      else:
        xz = np.array( [int(M['m10'] / M['m00']),
                        800-int(M['m01'] / M['m00']) ] )

    return[yz, xz]

  def jointMovement(self):
    pi = math.pi
    t = rospy.get_time() - self.initial_time
    j2 = (pi/3.0) * math.sin((pi/15.0) * t)
    j3 = (pi/3.0) * math.sin((pi/18.0) * t)
    j4 = (pi/3.0) * math.sin((pi/20.0) * t)
    return [j2, j3, j4]

  def get3Dim(self, data):
    yz = data[1]
    xz = data[0]
    return np.array([yz[0], xz[0], (xz[1] + yz[1])/2])
    # return [X, Y, Z] of each point

  def calcJointAngles(self, data):

    baseFrame = [0,0,0]

    yellow = baseFrame - data[0]
    blue = baseFrame - data[1]
    green = baseFrame - data[2]
    red = baseFrame - data[3]
    # J1 rotates around the z axis, so we measure the YZ points
    j1 = self.zAngle(blue - yellow)

    J1X = np.arctan2(yellow[1] - blue[1], yellow[2] - blue[2])
    J1Y = np.arctan2(yellow[0] - blue[0], yellow[2] - blue[2])
    J1Z = np.arctan2(yellow[0] - blue[0], yellow[1] - blue[1])

    j1 = np.arctan2(yellow[0] - blue[0], yellow[1] - blue[0])
    # J2 rotates around the X axis
    #j2 = self.xAngle(green - blue) - j1

    j2 = - (np.arctan2(blue[1] - green[1], blue[2] - green[2]) - J1X)
    #temp = j2 < 0
    #j2 = np.abs(j2) + 0.1
    #if(temp): j2 * -1
    #print("J3 testing: J3 =" + str((np.arctan2(blue[0] - green[0], blue[2] - green[2]))) + ", J1Y: "+ str(J1Y))
    print("Green Testing: " + str(green) + "  --- j3: " +  str((-1 * np.abs(np.arctan2(blue[0] - green[0], blue[2] - green[2]))) + J1Y));
    j3 =  (-1 * np.abs(np.arctan2(blue[0] - green[0], blue[2] - green[2]))) + J1Y
    # was getting j3 to be constantly slightly too big, offset just to make it more accurat e
    j3 = np.abs(j3) - 0.1
    # getting a weird bug that makes j3 go nuts when joint 3 is on the left of the base
    # fixed by getting it to just ignore the issue, act like it's always on the left in the actual calculation but check
    # to see if it's on the left, if it is then just times the calculated angle by -1
    if(green[0] > yellow[0]):
      j3 = -1 * j3


    j4 = np.abs((np.arctan2(green[1] - red[1], green[2] - red[2]))) - J1X - j2

    print("calc:     J2:" + str(np.round(1000 * j2) /1000) + ",  J3:" + str(np.round(1000 * j3) /1000) + ",  J4:" + str(np.round(1000 * j4) /1000))

  def pixel2Meter(self, circle1, circle2):
    return 2.5 / np.sqrt(np.sum((circle1 - circle2)**2))

  def xAngle(self, vector):
    return math.acos(np.dot([0,0], [vector[1], vector[2]]));

  def yAngle(self, vector):
    return math.acos(np.dot([0,0], [vector[1], vector[2]]));

  def zAngle(self, vector):
    return math.acos(np.dot([0,0], [vector[0], vector[1]]));

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


