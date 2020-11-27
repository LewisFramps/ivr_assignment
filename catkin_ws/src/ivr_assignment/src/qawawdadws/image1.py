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
        # if self.questionWanted = 2, image.py returns answers for 2.1 & 2.2
        # if self.questionWanted = 3.1, image.py returns answers for 3.1
        # if self.questionWanted = 3.2, image.py returns answers for 3.2
        # if self.questionWanted = 4, image.py returns answers for 4
        # Note you'll need to restart ROS each time due to some angles not changing between questions.
        self.questionWanted = 4;

        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera1 to a topic named image_topic1
        self.image_pub1 = rospy.Publisher("image_topic1", Image, queue_size=1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1)

        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

        self.robot_joint1Vision_pub = rospy.Publisher("/robot/joints1_estimation", Float64, queue_size=10)
        self.robot_joint2Vision_pub = rospy.Publisher("/robot/joints2_estimation", Float64, queue_size=10)
        self.robot_joint3Vision_pub = rospy.Publisher("/robot/joints3_estimation", Float64, queue_size=10)
        self.robot_joint4Vision_pub = rospy.Publisher("/robot/joints4_estimation", Float64, queue_size=10)

        self.robot_prevJoint1Val_pub = rospy.Publisher("/robot/Joint1Val", Float64, queue_size=10)
        self.robot_prevJoint2Val_pub = rospy.Publisher("/robot/Joint2Val", Float64, queue_size=10)
        self.robot_prevJoint3Val_pub = rospy.Publisher("/robot/Joint3Val", Float64, queue_size=10)
        self.robot_prevJoint4Val_pub = rospy.Publisher("/robot/Joint4Val", Float64, queue_size=10)

        self.robot_targetX_pup = rospy.Publisher("/robot/targetX", Float64, queue_size=10)
        self.robot_targetY_pup = rospy.Publisher("/robot/targetY", Float64, queue_size=10)
        self.robot_targetZ_pup = rospy.Publisher("/robot/targetZ", Float64, queue_size=10)

        self.robot_redX_pub = rospy.Publisher("/robot/redXVision", Float64, queue_size=10)
        self.robot_redY_pub = rospy.Publisher("/robot/redYVision", Float64, queue_size=10)
        self.robot_redZ_pub = rospy.Publisher("/robot/redZVision", Float64, queue_size=10)

        self.priorJ1 = None

        # colour_closest in format [Closest colour in YZ, Closest colour in XZ]
        # Used in case we can't see target colour, use the closest objects coordinates instead
        # Initialise to centre points just to clear any unforeseen errors.
        self.redClosest = np.array([[0.1, 0.1], [0.1, 0.1]])
        self.greenClosest = np.array([[0.1, 0.1], [0.1, 0.1]])
        self.blueClosest = np.array([[0.1, 0.1], [0.1, 0.1]])
        self.YellowClosest = np.array([[0.1, 0.1], [0.1, 0.1]])
        self.sphereClosest = np.array([[0.1, 0.1], [0.1, 0.1]])
        self.cubeClosest = np.array([[0.1, 0.1], [0.1, 0.1]])

        self.joint1 = Float64()
        self.joint2 = Float64()
        self.joint3 = Float64()
        self.joint4 = Float64()

        self.joint1Angle = 0.0
        self.joint2Angle = 0.0
        self.joint3Angle = 0.0
        self.joint4Angle = 0.0

        self.calcJoint1 = Float64()
        self.calcJoint2 = Float64()
        self.calcJoint3 = Float64()
        self.calcJoint4 = Float64()

        self.previousJoint1 = Float64()
        self.previousJoint2 = Float64()
        self.previousJoint3 = Float64()
        self.previousJoint4 = Float64()

        self.pixToMet = 0.05
        self.looped = False

        self.targetX = Float64()
        self.targetY = Float64()
        self.targetZ = Float64()

        self.redX = Float64()
        self.redY = Float64()
        self.redZ = Float64()

        # 3.2 stuff
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        self.error = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0, 0.0], dtype='float64')

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        self.cv_image2 = None
        self.initial_time = rospy.get_time()

        self.priorJ1Vector = None
        self.priorJ2Vector = None
        self.priorJ3Vector = None
        self.priorJ4Vector = None

    # Recieve data from camera 1, process it, and publish
    def callback1(self, data):
        # Recieve the image
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Uncomment if you want to save the image
        # cv2.imwrite('image_copy.png', cv_image)

        # get colour masks
        # colourMasks in format [redMask(YZ), redMask(XZ)]
        redMask = [self.get_red_mask(self.cv_image1), self.get_red_mask(self.cv_image2)]
        blueMask = [self.get_blue_mask(self.cv_image1), self.get_blue_mask(self.cv_image2)]
        yellowMask = [self.get_yellow_mask(self.cv_image1), self.get_yellow_mask(self.cv_image2)]
        greenMask = [self.get_green_mask(self.cv_image1), self.get_green_mask(self.cv_image2)]
        orangeMask = [self.get_orange_mask(self.cv_image1), self.get_orange_mask(self.cv_image2)]

        # get the visibility of each colour. colVis = [is seen in YZ, is seen in XZ]
        redVis = self.canSee(redMask)
        blueVis = self.canSee(blueMask)
        yellowVis = self.canSee(yellowMask)
        greenVis = self.canSee(greenMask)
        orangeVis = self.canSee(orangeMask)

        visibility = [yellowVis, blueVis, greenVis, redVis, orangeVis]

        redPixCentres = self.get_pixel_centre(redMask, redVis, self.redClosest)
        bluePixCentres = self.get_pixel_centre(blueMask, blueVis, self.blueClosest)
        yellowPixCentres = self.get_pixel_centre(yellowMask, yellowVis, self.YellowClosest)
        greenPixCentres = self.get_pixel_centre(greenMask, greenVis, self.greenClosest)

        orangePixCentresYZ = self.orangeObjectsPixelPos(orangeMask[0], 0)
        orangePixCentresXZ = self.orangeObjectsPixelPos(orangeMask[1], 1)

        targetPixCentres = np.array([orangePixCentresYZ[0], orangePixCentresXZ[0]])
        cubePixCentres = np.array([orangePixCentresYZ[1], orangePixCentresXZ[1]])

        pixelPoints = [yellowPixCentres, bluePixCentres, greenPixCentres, redPixCentres, targetPixCentres,
                       cubePixCentres]

        # get pixel to meter ratio only once, do this before any joints have moved.
        # This would break occasionally, it would always return 0.05 +- 0.01
        # So i've just hardcoded it, if you want to make sure it works remove the '&& False Is True'
        self.pixToMet = 0.05
        if not self.looped and False == True:
            self.pixToMet = self.pixel2Meter((yellowPixCentres[0] + yellowPixCentres[1]) / 2,
                                             (bluePixCentres[0] + bluePixCentres[1]) / 2)
            self.looped = True

        # Get 2D object locations in meters on the YZ and the XZ
        red2D = np.array([self.pixToMet * redPixCentres[0], self.pixToMet * redPixCentres[1]])
        blue2D = np.array([self.pixToMet * bluePixCentres[0], self.pixToMet * bluePixCentres[1]])
        yellow2D = np.array([self.pixToMet * yellowPixCentres[0], self.pixToMet * yellowPixCentres[1]])
        green2D = np.array([self.pixToMet * greenPixCentres[0], self.pixToMet * greenPixCentres[1]])
        target2D = np.array([self.pixToMet * targetPixCentres[0], self.pixToMet * targetPixCentres[1]])
        cube2D = np.array([self.pixToMet * cubePixCentres[0], self.pixToMet * cubePixCentres[1]])

        points2D = [red2D, blue2D, yellow2D, green2D, target2D, cube2D]

        # get 3D coordinates in meters for each object
        red3D = self.get3Dim(red2D)
        blue3D = self.get3Dim(blue2D)
        yellow3D = self.get3Dim(yellow2D)
        green3D = self.get3Dim(green2D)
        target3D = self.get3Dim(target2D)
        cube3D = self.get3Dim(cube2D)

        self.deltaJ1 = 1
        self.deltaJ2 = 1
        self.deltaJ3 = 1
        self.deltaJ4 = 1

        # Debugging images, delete this later!
        XY = np.ones((800, 800, 3), np.uint8)
        XY[800 - int(yellow3D[1] / self.pixToMet)][int(yellow3D[0] / self.pixToMet)] = (50, 255, 255)
        XY[800 - int(red3D[1] / self.pixToMet)][int(red3D[0] / self.pixToMet)] = (50, 50, 255)
        XY[800 - int(green3D[1] / self.pixToMet)][int(green3D[0] / self.pixToMet)] = (50, 255, 50)
        XY[800 - int(blue3D[1] / self.pixToMet)][int(blue3D[0] / self.pixToMet)] = (255, 50, 50)
        XY[800 - int(target3D[1] / self.pixToMet)][int(target3D[0] / self.pixToMet)] = (0, 165, 255)

        YZ = np.ones((800, 800, 3), np.uint8)
        YZ[800 - int(yellow3D[2] / self.pixToMet)][int(yellow3D[1] / self.pixToMet)] = (50, 255, 255)
        YZ[800 - int(red3D[2] / self.pixToMet)][int(red3D[1] / self.pixToMet)] = (50, 50, 255)
        YZ[800 - int(green3D[2] / self.pixToMet)][int(green3D[1] / self.pixToMet)] = (50, 255, 50)
        YZ[800 - int(blue3D[2] / self.pixToMet)][int(blue3D[1] / self.pixToMet)] = (255, 50, 50)
        YZ[800 - int(target3D[2] / self.pixToMet)][int(target3D[1] / self.pixToMet)] = (0, 165, 255)

        XZ = np.ones((800, 800, 3), np.uint8)
        XZ[800 - int(yellow3D[2] / self.pixToMet)][int(yellow3D[0] / self.pixToMet)] = (50, 255, 255)
        XZ[800 - int(red3D[2] / self.pixToMet)][int(red3D[0] / self.pixToMet)] = (50, 50, 255)
        XZ[800 - int(green3D[2] / self.pixToMet)][int(green3D[0] / self.pixToMet)] = (50, 255, 50)
        XZ[800 - int(blue3D[2] / self.pixToMet)][int(blue3D[0] / self.pixToMet)] = (255, 50, 50)
        XZ[800 - int(target3D[2] / self.pixToMet)][int(target3D[0] / self.pixToMet)] = (0, 165, 255)

        xyim = cv2.imshow("XY", XY)
        yzim = cv2.imshow("YZ", YZ)
        xzim = cv2.imshow("XZ", XZ)

        calcJoints = self.calcJointAngles([yellow3D, blue3D, green3D, red3D])

        self.calcJoint1 = calcJoints[0]
        self.calcJoint2 = calcJoints[1]
        self.calcJoint3 = calcJoints[2]
        self.calcJoint4 = calcJoints[3]

        self.setClosestPoints(points2D, visibility)

        # Get joint values that we're working on
        self.previousJoint1 = self.joint1
        self.previousJoint2 = self.joint2
        self.previousJoint3 = self.joint3
        self.previousJoint4 = self.joint4

        targetInBaseFrame = target3D - yellow3D
        self.targetX = 0.9 * targetInBaseFrame[0]
        self.targetY = 0.9 * targetInBaseFrame[1]
        self.targetZ = targetInBaseFrame[2]
        print("targets! " + str(targetInBaseFrame))

        RedInBaseFrame = red3D - yellow3D
        RedInBaseFrame = self.clampMagnitude(RedInBaseFrame, 9)
        self.redX = RedInBaseFrame[0]
        self.redY = RedInBaseFrame[1]
        self.redZ = RedInBaseFrame[2]


        joints = self.jointMovement()
        # Update joint angles
        if self.questionWanted == 2:
          self.joint1 = 0
          self.joint2 = joints[1]
          self.joint3 = joints[2]
          self.joint4 = joints[3]

        if self.questionWanted == 4:
          self.joint1 = joints[0]
          self.joint2 = joints[1]
          self.joint3 = joints[2]
          self.joint4 = joints[3]

        self.joint1Angle = self.joint1
        self.joint1Angle = self.joint2
        self.joint1Angle = self.joint3
        self.joint1Angle = self.joint4

        if self.questionWanted == 3.2:
          newAngles = self.closed_control(targetInBaseFrame)
          self.joint1 = newAngles[0]
          self.joint2 = newAngles[1]
          self.joint3 = newAngles[2]
          self.joint4 = newAngles[3]



        self.previousJoint1 = self.joint1
        self.previousJoint2 = self.joint2
        self.previousJoint3 = self.joint3
        self.previousJoint4 = self.joint4

        im1 = cv2.imshow('yzImage', self.cv_image1)
        im2 = cv2.imshow('xz,Image', self.cv_image2)
        if self.questionWanted == 3.1:
          self.joint1 = 15 * np.pi/180
          self.joint2 = 30 * np.pi/180
          self.joint3 = 60 * np.pi/180
          self.joint4 = 90 * np.pi/180
          print("FK calc: " + str(self.calc_end_effector_pos([self.joint1, self.joint2, self.joint3, self.joint4])))
          print("Vision: " + str(RedInBaseFrame[0]) + " , " + str(RedInBaseFrame[1]) + " , " + str(RedInBaseFrame[2]))

        # put this in 'callback' to continually update joint angles
        newAngles = self.closed_control(targetInBaseFrame)
        self.joint1 = newAngles[0]
        self.joint2 = newAngles[1]
        self.joint3 = newAngles[2]
        self.joint4 = newAngles[3]


        cv2.waitKey(1)
        # Publish the results
        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))

            self.robot_joint1_pub.publish(self.joint1)
            self.robot_joint2_pub.publish(self.joint2)
            self.robot_joint3_pub.publish(self.joint3)
            self.robot_joint4_pub.publish(self.joint4)

            self.robot_joint1Vision_pub.publish(self.calcJoint1)
            self.robot_joint2Vision_pub.publish(self.calcJoint2)
            self.robot_joint3Vision_pub.publish(self.calcJoint3)
            self.robot_joint4Vision_pub.publish(self.calcJoint4)

            self.robot_prevJoint1Val_pub.publish(self.previousJoint1)
            self.robot_prevJoint2Val_pub.publish(self.previousJoint2)
            self.robot_prevJoint3Val_pub.publish(self.previousJoint3)
            self.robot_prevJoint4Val_pub.publish(self.previousJoint4)

            self.robot_targetX_pup.publish(self.targetX)
            self.robot_targetY_pup.publish(self.targetY)
            self.robot_targetZ_pup.publish(self.targetZ)

            self.robot_redX_pub.publish(self.redX)
            self.robot_redY_pub.publish(self.redY)
            self.robot_redZ_pub.publish(self.redZ)


        except CvBridgeError as e:
            print(e)

    def clampMagnitude(self, v, mag):
        x, y, z = v
        n = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        f = min(n, mag) / n
        return np.array([x * f, y * f, z * f])

    def orangeObjectsPixelPos(self, image, index):
        targetPixelPos = np.array([-1, -1])
        cubePixelPos = np.array([-1, -1])
        targetContours = None
        cubeContours = None

        # Apply a orange mask and find contours of orange objects
        cont, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # if no orange objects are found we just set the targetsPixelPos to [0,0]. This is just for testing and I don't
        # think it can actually happen while running. If I have time I'll come back to this later
        if (len(cont) == 0):
            print("this really shouldn't happen")
            return np.array([self.sphereClosest / self.pixToMet, self.cubeClosest / self.pixToMet])

        # if objects are overlapping in view we just take the centre of both. This skew the results this is a rare
        # occurrence and if it does happen, it's usually for a tiny amount of time.
        elif (len(cont) == 1):
            targetContours = cont[0]
            cubeContours = cont[0]

        # If we find two objects, figure out which is the sphere and which is the cube and return their positions.
        # We know the circle contour should have a ton more vertices than the cube contour, so we compare those.
        elif (len(cont) == 2):
            if (len(cont[0]) > len(cont[1])):
                targetContours = cont[0]
                cubeContours = cont[1]
            else:
                targetContours = cont[1]
                cubeContours = cont[0]

        # this rarely happens but if it does then just return the last valid position#
        # only potential times this happens is when a link passes in front of a the orange objects
        elif len(cont) > 2:
            return np.array([self.sphereClosest[index] / self.pixToMet, self.cubeClosest[index] / self.pixToMet])

        # get the moments of the contours
        targM = cv2.moments(targetContours)
        cubeM = cv2.moments(cubeContours)

        # get centre position of each object
        if (targM['m00'] == 0):
            targetPixelPos = np.array([0, 0])
        else:
            targetPixelPos[0] = int(targM['m10'] / targM['m00'])
            targetPixelPos[1] = 800 - int(targM['m01'] / targM['m00'])

        if (cubeM['m00'] == 0):
            cubePixelPos = np.array([0, 0])
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

        kernal = np.ones((5, 5), np.uint8)
        return cv2.dilate(mask, kernal, iterations=3)

    def get_blue_mask(self, image):
        # Only grab blue pixels
        mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
        kernal = np.ones((5, 5), np.uint8)
        return cv2.dilate(mask, kernal, iterations=3)

    def get_yellow_mask(self, image):
        # Only grab yellow pixels
        hsvIm = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsvIm, (20, 100, 100), (30, 255, 255))
        kernal = np.ones((5, 5), np.uint8)
        return cv2.dilate(mask, kernal, iterations=3)

    def get_green_mask(self, image):
        # Only grab green pixels
        mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
        kernal = np.ones((5, 5), np.uint8)
        return cv2.dilate(mask, kernal, iterations=3)

    def get_orange_mask(self, image):
        # Only grab orange pixels
        hsvIm = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsvIm, (1, 150, 80), (18, 255, 255))
        kernal = np.ones((5, 5), np.uint8)
        return cv2.dilate(mask, kernal, iterations=1)

    def get_pixel_centre(self, mask, seen, closest):
        yz = None
        xz = None

        # Find mask pixel centre in yz plane

        # if the colour doesn't show in the plane, take the colour that was closest to it before it disappeared
        if not seen[0]:
            yz = closest[0] / self.pixToMet
        else:
            M = cv2.moments(mask[0])
            if M['m00'] == 0:
                yz = np.array([0, 0])
            else:
                yz = np.array([int(M['m10'] / M['m00']),
                               800 - int(M['m01'] / M['m00'])])

        # Find mask pixel centre in xz plane
        # if the colour doesn't show in the plane, take the colour that was closest to it before it disappeared
        if not seen[1]:
            xz = closest[1] / self.pixToMet
        else:
            M = cv2.moments(mask[1])
            if M['m00'] == 0:
                xz = np.array([0, 0])
            else:
                xz = np.array([int(M['m10'] / M['m00']),
                               800 - int(M['m01'] / M['m00'])])

        return [yz, xz]

    def jointMovement(self):
        pi = math.pi
        t = rospy.get_time() - self.initial_time
        j1 = (pi) * math.sin((pi/15) * t)
        j2 = (pi / 2.0) * math.sin((pi / 15.0) * t)
        j3 = (pi / 2.0) * math.sin((pi / 18.0) * t)
        j4 = (pi / 3.0) * math.sin((pi / 20.0) * t)
        return [j1, j2, j3, j4]

    def get3Dim(self, data):
        yz = data[1]
        xz = data[0]
        return np.array([yz[0], xz[0], (xz[1] + yz[1]) / 2])
        # return [X, Y, Z] of each point

    def calcJointAngles(self, data):

        baseFrame = [0, 0, 0]

        yellow = baseFrame - data[0]
        blue = baseFrame - data[1]
        green = baseFrame - data[2]
        red = baseFrame - data[3]

        yellowToBlue = blue - yellow
        blueToGreen = green - blue
        greenToRed = red - green



        # J1 rotates around the z axis, so we measure the YZ points
        print("I got here")
        ytb = [yellowToBlue[0], yellowToBlue[1]] / np.linalg.norm([yellowToBlue[0], yellowToBlue[1]])
        self.calcNewJ1 = 0
        j1 = 0
        if(self.priorJ1 is None or math.isnan(self.priorJ1)):
          print("!")
          self.calcNewJ1 = np.dot(ytb, [0,1]) * -np.pi
          self.priorJ1 = self.calcNewJ1
          if(math.isnan(self.priorJ1)):
              self.priorJ1 = 0
        else:
          self.calcNewJ1 = np.dot(ytb, [0,1]) * -np.pi
          print("calcNew: " + str(self.calcNewJ1))
          changeInJ1 =  self.calcNewJ1 - self.priorJ1
          print(self.priorJ1)
          changeInJ1 = self.limitVal(changeInJ1, -0.15, 0.15)
          self.calcNewJ1 = self.priorJ1 + changeInJ1
          self.priorJ1 = self.calcNewJ1
          j1 = self.calcNewJ1
        if self.questionWanted != 4:
          j1 = 0

        ZRotationMatrix = np.array([[np.cos(j1), -np.sin(j1), 0], [np.sin(j1), np.cos(j1), 0], [0, 0, 1] ])

        if math.isnan(j1):
          print("kill me")

        # Get joint 3s angle by checking its angle in the YZ plane

        if(self.questionWanted == 4):
          blueToGreen = np.transpose(np.matmul(ZRotationMatrix, np.transpose(blueToGreen)))
          greenToRed =  np.transpose(np.matmul(ZRotationMatrix, np.transpose(greenToRed)))

        j2 = np.arccos(blueToGreen[1] / (np.sqrt(blueToGreen[1] ** 2 + blueToGreen[2] ** 2))) + np.pi / 2
        if(self.questionWanted == 4):
          j2 = -np.arccos(blueToGreen[1] / (np.sqrt(blueToGreen[1] ** 2 + blueToGreen[2] ** 2))) + np.pi / 2

        j3 = 0.85 * (np.arccos(blueToGreen[0] / (np.sqrt(blueToGreen[0] ** 2 + blueToGreen[2] ** 2))) - np.pi / 2)
        if(self.questionWanted == 4):
          j3= 0.85 * (np.arccos(blueToGreen[0] / (np.sqrt(blueToGreen[0] ** 2 + blueToGreen[2] ** 2))) - np.pi / 2)

        j4 = -np.arccos(greenToRed[1] / (np.sqrt(greenToRed[1] ** 2 + greenToRed[2] ** 2))) + np.pi / 2 - j2
        if(self.questionWanted == 4):
            j4 = (np.arccos(greenToRed[1] / (np.sqrt(greenToRed[1] ** 2 + greenToRed[2] ** 2))) - np.pi/2 - j2) * 0.5

        return [self.limitVal(j1, -np.pi, np.pi), self.limitVal(j2, -np.pi / 2, np.pi / 2), self.limitVal(j3, -np.pi / 2, np.pi / 2),
                self.limitVal(j4, -np.pi / 3, np.pi / 3)]

    def limitVal(self, val, min, max):
        if val > max:
            return max
        elif val < min:
            return min
        return val

    def pixel2Meter(self, circle1, circle2):
        return 2.5 / np.sqrt(np.sum((circle1 - circle2) ** 2))

    def xAngle(self, vector):
        return math.acos(np.dot([0, 0], [vector[1], vector[2]]));

    def yAngle(self, vector):
        return math.acos(np.dot([0, 0], [vector[1], vector[2]]));

    def zAngle(self, vector):
        return math.acos(np.dot([0, 0], [vector[0], vector[1]]));

    def getClosestPoint(self, points, ownPoint, index):
        closestPoint = None
        MinDist = 1000000;
        for point in points:
            Dist = self.sqrdist(point[index], ownPoint)
            if Dist < MinDist and Dist > 1:
                closestPoint = point[index]
                MinDist = Dist
        return closestPoint

    def setClosestPoints(self, points, visibility):
        # if we can see it, update it!
        # if you're reading this, I wrote this at like 1:30am and yeah I know it's horrible but I do not care at this point!
        # It's all being abstracted away in a function to be forgotten about anyways
        if visibility[0][0]: self.YellowClosest[0] = self.getClosestPoint(points, points[0][0], 0)
        if (visibility[0][1]): self.YellowClosest[1] = self.getClosestPoint(points, points[0][1], 1)

        if (visibility[1][0]): self.blueClosest[0] = self.getClosestPoint(points, points[1][0], 0)
        if (visibility[1][1]): self.blueClosest[1] = self.getClosestPoint(points, points[1][1], 1)

        if (visibility[2][0]): self.greenClosest[0] = self.getClosestPoint(points, points[2][0], 0)
        if (visibility[2][1]): self.greenClosest[1] = self.getClosestPoint(points, points[2][1], 1)

        if (visibility[3][0]): self.redClosest[0] = self.getClosestPoint(points, points[3][0], 0)
        if (visibility[3][1]): self.redClosest[1] = self.getClosestPoint(points, points[3][1], 1)

        if (visibility[4][0]): self.sphereClosest[0] = self.getClosestPoint(points, points[4][0], 0)
        if (visibility[4][1]): self.sphereClosest[1] = self.getClosestPoint(points, points[4][1], 1)

        if (visibility[4][0]): self.cubeClosest[0] = self.getClosestPoint(points, points[5][0], 0)
        if (visibility[4][1]): self.cubeClosest[1] = self.getClosestPoint(points, points[5][1], 1)

    def sqrdist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def jacobianCalc(self, jVal):
        # 3.2 stuff
        jVal0 = jVal
        jVal1 = jVal
        jVal2 = jVal
        jVal3 = jVal

        pi = np.pi
        hPi = pi / 2.0
        c1 = np.cos(jVal0 + hPi)
        c2 = np.cos(jVal1 + hPi)
        c3 = np.cos(jVal2)
        c4 = np.cos(jVal3)
        s1 = np.sin(jVal0 + hPi)
        s2 = np.sin(jVal1 + hPi)
        s3 = np.sin(jVal2)
        s4 = np.sin(jVal3)

        jacobian = np.array([[-3.5 * c2 * c3 * s1 + 3 * s2 * s4 * s1 + 3 * c4 * (-1 * c2 * c3 * c1 + s3 * c1) + 3.5 * s3 * c1, -3.5 * c1 * c3 * s2 - 3 * c1 * s4 * c2 - 3 * c4 * c1 * c3 * s2, 3.5 * s1 * c3 + 3 * c4 * (s1 * c3 - c1 * c2 * s3) - 3.5 * c1 * c2 * s3, -3 * c1 * s2 * c4 - 3 * s4 * (s1 * s3 + c1 * c2 * c3)],
                             [3.5 * c2 * c3 * c1 - 3 * s2 * s4 * c1 + 3 * c4 * (s3 * s1 + c2 * c3 * c1) + 3.5 * s3 * s1, -3.5 * s1 * c3 * s2 - 3 * s1 * s4 * c2 - 3 * c4 * s1 * c3 * s2, 3 * c4 * (-1 * s1 * c2 * s3 - c1 * c3)- 3.5 * s1 * c2 * s3 - 3.5 * c1 * c3,  -3 * s1 * s2 * c4 - 3 * s4 * (s1 * c2 * c3 - c1 * s3)],
                             [np.array(0), 3 * c3 * c4 * c2 + 3.5 * c3 * c2- 3 * s4 * s2, -3 * s2 * c4 * s3- 3.5 * s2 * s3, 3 * c2 * c4 - 3 * s2 * c3 * s4]])
        return jacobian

    def psuedoJacobianCalc(self, jacobian):
        jacobianTrans = np.transpose(jacobian)
        psuedoJacobian = np.dot(jacobianTrans, np.linalg.pinv(np.dot(jacobian, jacobianTrans)))
        return psuedoJacobian

    def cos(self, i):
        return np.cos(i)

    def sin(self, i):
        return np.sin(i)

    def getAMatrix(self, a, alpha, d, theta):
        test = np.matrix([[self.cos(theta), -self.sin(theta)*self.cos(alpha), self.sin(theta)*self.sin(alpha), a*self.cos(theta)], [self.sin(theta), self.cos(theta)*self.cos(alpha), -self.cos(theta)*self.sin(alpha), a*self.sin(theta)], [0, self.sin(alpha), self.cos(alpha), d], [0,0,0,1]])
        return test

    def calc_end_effector_pos(self, joints):
        # 3.2 stuff
        pi = np.pi

        hPi = pi / 2

        j1 = joints[0]
        j2 = joints[1]
        j3 = joints[2]
        j4 = joints[3]

        A10 = self.getAMatrix(0, hPi, 2.5, (j1 + hPi))
        A21 = self.getAMatrix(0, hPi, 0, (j2 + hPi))
        A32 = self.getAMatrix(3.5, -hPi, 0, j3)
        A43 = self.getAMatrix(3, 0, 0, j4)

        I = np.dot(A10, A21)
        J = np.dot(A32, A43)
        K = np.dot(I, J)
        effector_pos = np.array([K.item(0,3), K.item(1,3), K.item(2,3)])
        return effector_pos

    def closed_control(self, targetLocation):
        # P and D gain
        K_p = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        K_d = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])

        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        self.time_previous_step = cur_time

        # joint angles
        jointAngles = self.jointMovement()
        jointAngles = [self.previousJoint2, self.previousJoint3]
        q = np.array([self.joint1Angle, self.joint2Angle, self.joint3Angle, self.joint4Angle])

        # end effector position and sphere position
        pos = self.calc_end_effector_pos(q)
        pos_d = targetLocation

        # estimate error
        self.error_d = ((pos_d - pos) - self.error) / dt
        self.error = pos_d - pos

        # calc psuedo inverse of jacobian
        J_inv = self.psuedoJacobianCalc(self.jacobianCalc(q))

        # calculate angualr velocity of joints

        dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p, self.error.transpose())))

        # integrate to find control inputs
        q_d = q + (dt * dq_d)
        return q_d



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
