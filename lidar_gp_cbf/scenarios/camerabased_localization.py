import numpy as np
from math import atan2,degrees,atan
import cv2
from nebolab_experiment_setup import NebolabSetup as nebolab
# import nebolab_basic_setup as nebolab


DRAW_MARKER_CONTOUR = False

# and alternative class for ROS Twist message
class pose_class:
    def __init__(self, px=0., py=0., theta=0.):
        self.x, self.y, self.theta = px, py, theta
    
# OBJECT DETECTION 
#--------------------------------------------------------------------    
class localize_from_ceiling:
    def __init__(self, robot_num=4):
        # Set range for color detection (lower and upper)
        self.HSV_min = []
        self.HSV_max = []
        self.HSV_min += [ np.array([00, 80, 219], np.uint8) ] # red
        self.HSV_max += [ np.array([11, 255, 255], np.uint8) ]
        # self.HSV_min += [ np.array([101, 67, 224], np.uint8) ] # blue
        self.HSV_min += [ np.array([101, 67, 224], np.uint8) ] # blue
        self.HSV_max += [ np.array([137, 255, 255], np.uint8) ]
        self.HSV_min += [ np.array([76, 117, 195], np.uint8) ] # green
        self.HSV_max += [ np.array([90, 255, 255], np.uint8) ]  
        self.HSV_min += [ np.array([28, 71, 241], np.uint8) ] # orange
        self.HSV_max += [ np.array([48, 255, 255], np.uint8) ]
        # TODO: do some assert to make sure have same length

        # PARAMETER FOR MARKER DETECTION (Rectangle and Circle) 
        # - total number of pixels -
        self.rect_upper = 2400
        self.rect_lower = 1000
        self.circle_upper = 1000
        self.circle_lower = 150

        self.n = robot_num

    def compute_pose(self, image_rgb, contours):
        rect = []
        circ = []
        for c in contours:        
            # Compute the centroid of contour
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # Plot the contour centroid
            # cv2.drawContours(image_rgb, [c], -1, (0, 255, 0), 1)

            # Compute the Area of each contour
            area = cv2.contourArea(c)
            # record all area that can be specified as rectangle or circle
            if self.rect_upper > area > self.rect_lower: 
                if DRAW_MARKER_CONTOUR:
                    cv2.circle(image_rgb,(cx,cy),2,(0,0,0),1)
                    cv2.drawContours(image_rgb, [c], -1, (255, 0, 0), 1)  
                rect += [(cx,cy)]
            elif self.circle_upper > area > self.circle_lower : 
                if DRAW_MARKER_CONTOUR:
                    cv2.circle(image_rgb,(cx,cy),2,(0,0,0),1)
                    cv2.drawContours(image_rgb, [c], -1, (0, 255, 0), 1)  
                circ += [(cx,cy)]
            # else: should skip this contour

        # For now just use the last value of data if not empty
        # TODO: check valid distance between markers
        rect_pxl = rect[-1] if len(rect) > 0 else None
        circ_pxl = circ[-1] if len(circ) > 0 else None

        if rect_pxl is not None and circ_pxl is not None: 
            pose_center = pose_class()
            pose_ahead = pose_class()

            # Change the marker position into meter, to make the flow more understandable
            p_rectx, p_recty = nebolab.pos_pxl2m(rect_pxl[0], rect_pxl[1])
            p_circx, p_circy = nebolab.pos_pxl2m(circ_pxl[0], circ_pxl[1])

            p_angle = atan2( (p_recty-p_circy),(p_rectx-p_circx)) + np.pi/2 # shift the angle 90deg
            # Compute centroid position between the two marker
            p_centx = (p_rectx + p_circx)/2
            p_centy = (p_recty + p_circy)/2        
            # Compute center point of wheel 
            p_cWheelx = p_centx + nebolab.TB_OFFSET_CENTER_WHEEL*np.cos(p_angle)
            p_cWheely = p_centy + nebolab.TB_OFFSET_CENTER_WHEEL*np.sin(p_angle)
            # Compute point Ahead position
            p_pAheadx = p_cWheelx + nebolab.TB_L_SI2UNI*np.cos(p_angle)
            p_pAheady = p_cWheely + nebolab.TB_L_SI2UNI*np.sin(p_angle)

            # Return the pose to be published
            pose_center.x = p_centx
            pose_center.y = p_centy 
            pose_center.theta = p_angle

            pose_ahead.x = p_pAheadx
            pose_ahead.y = p_pAheady 
            pose_ahead.theta = p_angle

            return pose_center, pose_ahead
        else: return None, None # Not valid data


    def localize_all_robots(self, image_rgb):
        kernel = np.ones((5,5),np.float32)/25
        #image_rgb = imutils.resize(image_rgb,width=1080,height=720)
        image_rgb = cv2.GaussianBlur(image_rgb,(3,3),0)

        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)
        imgray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # Morphological Transform, Dilation
        # for each color and bitwise_and operator
        # between imageFrame and mask determines
        # to detect only that particular color
        self.poses_center = [] # [r_pose, b_pose, g_pose, o_pose]
        self.poses_ahead = [] # [r_pose, b_pose, g_pose, o_pose]
        kernal = np.ones((10, 10), "uint8")
        for i in range(self.n):
            temp_mask = cv2.inRange(hsv, self.HSV_min[i], self.HSV_max[i])
            mask = cv2.dilate(temp_mask, kernal)
            # Compute contours from masked color
            cont, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
            
            # MAIN COMPUTATION FROM DETECTED CONTOUR INTO POSITION AND POSE
            # ------------------------------------------------------------------
            # Compute the pose (position and angle) from detected contour
            pose_center, pose_ahead = self.compute_pose( image_rgb, cont)
            self.poses_center += [pose_center]
            self.poses_ahead += [pose_ahead]
        
        return self.poses_center, self.poses_ahead
        
    def draw_pose(self, image_raw):
        image_rgb = image_raw.copy()
        for i in range(self.n):
            if self.poses_center[i] is not None:
                cur_pcent = (nebolab.pos_m2pxl(self.poses_center[i].x, self.poses_center[i].y))
                cur_pahead = (nebolab.pos_m2pxl(self.poses_ahead[i].x, self.poses_ahead[i].y))
                # Draw the centroid
                cv2.circle(image_rgb, cur_pcent, 1,(255,255,255),2)
                # Draw arrow from center of wheel to point ahead position
                cv2.arrowedLine(image_rgb, cur_pcent, cur_pahead,(0,255,0),2,tipLength=0.2)
                #cv2.arrowedLine(image_rgb,(nebolab.pos_m2pxl(p_cWheelx,p_cWheely)),(nebolab.pos_m2pxl(p_pAheadx,p_pAheady)),(0,255,0),2,tipLength=0.2)
        
        return image_rgb
