import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import time
import numpy as np
import sys
import math
import keyboard
import os

# selectie numar camera (0 - camera interna a laptopului, 1,2 - camera externa)
CAM_NUMBER = 0

class AppMonitoring:
    def __init__(self, file_with_camera_params):
        self.file_with_camera_params = file_with_camera_params
        self.init_camera_params()
        
    def init_camera_params(self):
        # loaded_params = np.load('calibrate_camera_parameters.npz', allow_pickle = True)
        loaded_params = np.load(self.file_with_camera_params, allow_pickle = True)
        self.cam_mtx = loaded_params['arr_0']
        self.cam_dist = loaded_params['arr_1']

    def undistort_image(self, original_image):
        # cod pentru rectificarea distorsionarii imaginii, pentru a obtine o imagine ne-distorsionata
        h,  w = original_image.shape[:2]
        # obtinerea unei versiuni ajustate a matricei de calibrare a camerei
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.cam_mtx, self.cam_dist, (w, h), 1, (w, h))
        
        # undistort - metoda folosita pentru initierea hartii de rectificare a imaginii
        # ultimul argument : factorul de scalare a hartilor de rectificare; factor mai mare -> rezolutie mai mare a hartilor && memorie mai multa
        mapx, mapy = cv2.initUndistortRectifyMap(self.cam_mtx, self.cam_dist, None, newcameramtx, (w, h), 5)

        # functia care aplica rectificarea distorsiunii asupra imaginii; utilizeaza si hartile generate anterior
        dst = cv2.remap(original_image, mapx, mapy, cv2.INTER_LINEAR)

        # crop the image
        # din moment ce imaginea initiala este distorsionata, dupa rectificarea distorsiunii, unii pixeli din partile extreme ale cadrului
        # dispar; se re-dimensioneaza imaginea
        x, y, w, h = roi
        image = dst[y:y+h, x:x+w]
        
        return image
    
    def start_application(self):
        raise NotImplementedError
    

class DeskMonitoring(AppMonitoring):
    def __init__(self, params_file):
        super().__init__(file_with_camera_params = params_file)
        self.run = None
        self.selected_side = 'right'
        self.recalibrate = None
        self.color = (0,0,255)
        self.back_calibrated = False
        self.leg_calibrated = False
        self.arm_calibrated = False
        self.head_calibrated = False
        self.max_height_hip_shoulder = 0
        # self.min_raport_correct_position_back = 0.94
        self.min_raport_correct_position_back = 0.93
        self.step = 1
        self.default_leg_angle = 0
        self.default_arm_angle = 0
        self.default_head_angle = 0

        self.delta_ts = 0
        self.time_start = time.time()
        
    def start_application(self):
        # define a video capture object
        self.vid = cv2.VideoCapture(CAM_NUMBER)

        # Initialize mediapipe pose class.
        self.mp_pose = mp.solutions.pose

        # Setup the Pose function for images - independently for the images standalone processing.
        self.pose_image = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # Initialize mediapipe drawing class - to draw the landmarks points.
        self.mp_drawing = mp.solutions.drawing_utils
        self.run = True
        while(self.run):
            try:
                # Capture the video frame
                ret, frame = self.vid.read()
                
                undistorted_image = self.undistort_image(frame)

                self.detectPose(undistorted_image, self.pose_image, draw=True, display=True)
                # the 'q' button is set as the
                # quitting button you may use any
                # desired button of your choice
            except:
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap object
        self.vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

    def change_side(self, side):
        self.selected_side = side

    def draw_border(self, image):
        if self.back_calibrated and self.leg_calibrated and self.arm_calibrated:
            border_size = 7
            image = cv2.copyMakeBorder(
                image,
                top=border_size,
                bottom=border_size,
                left=border_size,
                right=border_size,
                borderType=cv2.BORDER_CONSTANT,
                value=[self.color[0], self.color[1], self.color[2]]
            )
        # import pdb;pdb.set_trace()
        return image

    def text_on_screen(self, image, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (50, 50)
        # fontScale
        fontScale = 0.6
        # Line thickness of 2 px
        thickness = 2
        # time.sleep(0.5)
        image_result = cv2.putText(image, text, org, font, fontScale, self.color, thickness, cv2.LINE_AA)
        return image_result

    def detectPose(self, image_pose, pose, draw=False, display=False):
        original_image = image_pose.copy()

        image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)

        resultant = pose.process(image_in_RGB)
        needed_landmarks = sorted([
            self.mp_pose.PoseLandmark.NOSE.value,
            # self.mp_pose.PoseLandmark.LEFT_EYE_INNER.value,
            # self.mp_pose.PoseLandmark.LEFT_EYE.value,
            # self.mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
            # self.mp_pose.PoseLandmark.RIGHT_EYE_INNER.value,
            # self.mp_pose.PoseLandmark.RIGHT_EYE.value,
            # self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
            # self.mp_pose.PoseLandmark.LEFT_EAR.value,
            self.mp_pose.PoseLandmark.RIGHT_EAR.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            ])
        connections = [(needed_landmarks.index(x[0]), needed_landmarks.index(x[1])) for x in self.mp_pose.POSE_CONNECTIONS if x[0] in needed_landmarks and x[1] in needed_landmarks]

        # print(needed_landmarks)
        if resultant.pose_landmarks and draw:
            landmarks = resultant.pose_landmarks.landmark
            landmarks_subset = [x for (idx, x) in enumerate(resultant.pose_landmarks.landmark) if idx in needed_landmarks]
            landmark_subset = landmark_pb2.NormalizedLandmarkList(landmark = landmarks_subset)   
            # Calculate angle

        points = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                landmarks[self.mp_pose.PoseLandmark.NOSE.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value],

                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
                landmarks[self.mp_pose.PoseLandmark.NOSE.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value],

                ]

        if self.selected_side == 'right':
            offset = 0
        else:
            offset = 8

        # angle_arm = calculate_arm_angle(points[0], points[1], points[2])
        leg_angle = calculate_angle_3_points(points[3 + offset], points[4 + offset], points[5 + offset])
        # print(leg_angle)
        arm_angle = calculate_angle_3_points(points[0 + offset], points[1 + offset], points[2 + offset])
        # print(arm_angle)
        head_angle = calculate_angle_3_points(points[7 + offset], points[0 + offset], points[3 + offset])
        
        current_distance = calculate_distance_2_points(points[0 + offset], points[3 + offset])

        if not self.back_calibrated:
            original_image = self.text_on_screen(original_image, "Tineti spatele drept pentru calibrare.")
        elif not self.leg_calibrated:
            self.color = (0,0,255)
            original_image = self.text_on_screen(original_image, "Asezati picioarele perpendicular pe sol.")
        elif not self.arm_calibrated:
            self.color = (0,0,255)
            original_image = self.text_on_screen(original_image, "Asezati mainile in pozitia obisnuita, dar corecta.")
        elif not self.head_calibrated:
            self.color = (0,0,255)
            original_image = self.text_on_screen(original_image, "Uitati-va in centrul ecranului.")
        else:
            current_raport = current_distance / self.max_height_hip_shoulder
            leg_angle_error = calculate_angle_error(self.default_leg_angle, leg_angle)
            arm_angle_error = calculate_angle_error(self.default_arm_angle, arm_angle)
            head_angle_error = calculate_angle_error(self.default_head_angle, head_angle)

            if current_raport < self.min_raport_correct_position_back:
                self.color = (255, 0, 0)
                original_image = self.text_on_screen(original_image, "Indreapta spatele")
            elif leg_angle_error > 0.1:
                self.color = (255, 0, 0)
                original_image = self.text_on_screen(original_image, "Tineti picioarele perpendiculare pe sol")
            elif arm_angle_error > 0.1:
                self.color = (255, 0, 0)
                original_image = self.text_on_screen(original_image, "Tineti mainile in pozitia corecta")
            elif head_angle_error > 0.02:
                self.color = (255, 0, 0)
                original_image = self.text_on_screen(original_image, "Indreapta capul")
            else:
                self.color = (0, 255, 0)
            original_image = self.draw_border(original_image)

        if  keyboard.is_pressed("enter"):
            self.delta_ts = time.time() - self.time_start
            # print('enter is pressed')
            if self.delta_ts > 1:
                if self.step <= 5:
                    self.step += 1
                    self.time_start = time.time()

            
        if self.recalibrate:
            self.recalibrate = False
            print('recalibration is starting')
            self.back_calibrated = False
            self.leg_calibrated = False
            self.arm_calibrated = False
            self.head_calibrated = False
            self.max_height_hip_shoulder = 0
            self.default_leg_angle = 0
            self.default_arm_angle = 0
            self.default_head_angle = 0
            self.step = 1
            self.color = (0,0,255)

        if self.step == 2:
            if self.max_height_hip_shoulder < current_distance:
                self.max_height_hip_shoulder = current_distance
            self.back_calibrated = True
        if self.step == 3:
            self.default_leg_angle = leg_angle
            self.leg_calibrated = True
        if self.step == 4:
            self.default_arm_angle = arm_angle
            self.arm_calibrated = True
        if self.step == 5:
            self.default_head_angle = head_angle
            self.head_calibrated = True
            self.step += 1


        self.mp_drawing.draw_landmarks(image=original_image, landmark_list=landmark_subset,
                                connections=connections,
                                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,0,255),
                                                                                thickness=6, circle_radius=1),
                                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0),
                                                                                thickness=3, circle_radius=1))

        if display:
            
            cv2.imshow('frame', original_image[:,:,::-1])
        else:        
            return original_image, resultant
        

class SquatMonitoring(AppMonitoring):
    def __init__(self, params_file):
        super().__init__(file_with_camera_params = params_file)
        self.run = None
        self.color = (0,0,255)
        self.selected_side = 'right'
        
        self.step = None
        self.start_leg_angle = 90
        self.start_hip_angle = 90
        self.start_back_angle = 90

        self.previous_leg_angle = 90
        self.previous_hip_angle = 90
        self.previous_back_angle = 90

        self.wanted_leg_angle = 55
        self.wanted_hip_angle = 180
        self.wanted_back_angle = 60

        self.squats_count = 0
        self.error_tolerance = 10

        self.delta_ts = 0
        self.time_start = time.time()
    
    def start_application(self):
        # define a video capture object
        self.vid = cv2.VideoCapture(CAM_NUMBER)

        # Initialize mediapipe pose class.
        self.mp_pose = mp.solutions.pose

        # Setup the Pose function for images - independently for the images standalone processing.
        self.pose_image = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # Initialize mediapipe drawing class - to draw the landmarks points.
        self.mp_drawing = mp.solutions.drawing_utils
        self.run = True
        while(self.run):
            try:
                # Capture the video frame
                ret, frame = self.vid.read()

                self.detectPose(frame, self.pose_image, draw=True, display=True)
                # the 'q' button is set as the
                # quitting button you may use any
                # desired button of your choice
            except:
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap object
        self.vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

    def change_side(self, side):
        self.selected_side = side

    def draw_border(self, image):
        border_size = 7
        image = cv2.copyMakeBorder(
            image,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[self.color[0], self.color[1], self.color[2]]
        )
        return image

    def text_on_screen(self, image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (50, 50)
        # fontScale
        fontScale = 0.6
        # Line thickness of 2 px
        thickness = 2
        image_result = cv2.putText(image, str(self.squats_count), org, font, fontScale, (0,0,0), thickness, cv2.LINE_AA)
        return image_result

    def detectPose(self, image_pose, pose, draw=False, display=False):
        original_image = image_pose.copy()
        if self.selected_side == 'left':
            original_image = cv2.flip(original_image, 1)

        image_in_RGB = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        resultant = pose.process(image_in_RGB)
        needed_landmarks = sorted([
            self.mp_pose.PoseLandmark.NOSE.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            ])
        connections = [(needed_landmarks.index(x[0]), needed_landmarks.index(x[1])) for x in self.mp_pose.POSE_CONNECTIONS if x[0] in needed_landmarks and x[1] in needed_landmarks]

        # print(needed_landmarks)
        if resultant.pose_landmarks and draw:
            landmarks = resultant.pose_landmarks.landmark
            landmarks_subset = [x for (idx, x) in enumerate(resultant.pose_landmarks.landmark) if idx in needed_landmarks]
            landmark_subset = landmark_pb2.NormalizedLandmarkList(landmark = landmarks_subset)   
            # Calculate angle

        points = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value],

                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
                ]

        if self.selected_side == 'right':
            offset = 0
        else:
            offset = 6

        leg_angle = angleOf(points[4 + offset], points[5 + offset])
        hip_angle = angleOf(points[3 + offset], points[4 + offset])
        back_angle = angleOf(points[0 + offset], points[3 + offset])

        if self.step is None:
            if self.start_leg_angle - self.error_tolerance <= leg_angle <= self.start_leg_angle + self.error_tolerance and \
                self.start_hip_angle - self.error_tolerance <= hip_angle <= self.start_hip_angle + self.error_tolerance and \
                self.start_back_angle - self.error_tolerance <= back_angle <= self.start_back_angle + self.error_tolerance:
                print('going to step 1')
                self.step = 1
                self.previous_leg_angle = leg_angle
                self.previous_hip_angle = hip_angle
                self.previous_back_angle = back_angle
        elif self.step == 1:
            self.color = (0,255,0)
            if leg_angle > self.previous_leg_angle + self.error_tolerance or hip_angle < self.previous_hip_angle - self.error_tolerance or back_angle > self.previous_back_angle + self.error_tolerance:
                self.step = None
                print('executie gresita. du-te la pozitia initiala')
                self.color = (255,0,0)
                print('leg angle ', leg_angle, self.previous_leg_angle)
                print('hip angle ', hip_angle, self.previous_hip_angle)
                print('bak angle ', back_angle, self.previous_back_angle)
            elif self.wanted_leg_angle - self.error_tolerance <= leg_angle <= self.wanted_leg_angle + self.error_tolerance and self.wanted_hip_angle - self.error_tolerance <= hip_angle <= self.wanted_hip_angle + self.error_tolerance and self.wanted_back_angle - self.error_tolerance <= back_angle <= self.wanted_back_angle + self.error_tolerance:
                self.step = 2
                print('going to step 2')

            self.previous_leg_angle = leg_angle if leg_angle < self.previous_leg_angle else self.previous_leg_angle
            self.previous_hip_angle = hip_angle if hip_angle > self.previous_hip_angle else self.previous_hip_angle
            self.previous_back_angle = back_angle if back_angle < self.previous_back_angle else self.previous_back_angle
        else:
            if leg_angle < self.previous_leg_angle - self.error_tolerance or hip_angle > self.previous_hip_angle + self.error_tolerance or back_angle < self.previous_back_angle - self.error_tolerance:
                self.step = None
                self.color = (255,0,0)
                print('executie gresita. du-te la pozitia initiala')
                print('leg angle ', leg_angle)
                print('hip angle ', hip_angle)
                print('bak angle ', back_angle)

            elif self.start_leg_angle - self.error_tolerance <= leg_angle <= self.start_leg_angle + self.error_tolerance and self.start_hip_angle - self.error_tolerance <= hip_angle <= self.start_hip_angle + self.error_tolerance and self.start_back_angle - self.error_tolerance <= back_angle <= self.start_back_angle + self.error_tolerance:
                self.step = None
                self.squats_count += 1
                print(self.squats_count)
            self.previous_leg_angle = leg_angle if leg_angle > self.previous_leg_angle else self.previous_leg_angle
            self.previous_hip_angle = hip_angle if hip_angle < self.previous_hip_angle else self.previous_hip_angle
            self.previous_back_angle = back_angle if back_angle > self.previous_back_angle else self.previous_back_angle
            
        original_image = self.text_on_screen(original_image)
        original_image = self.draw_border(original_image)

        self.mp_drawing.draw_landmarks(image=original_image, landmark_list=landmark_subset,
                                connections=connections,
                                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,0,255),
                                                                                thickness=6, circle_radius=1),
                                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0),
                                                                                thickness=3, circle_radius=1))

        if display:
            
            cv2.imshow('frame', original_image[:,:,::-1])
        else:        
            return original_image, resultant


class CameraCalibration():
    def __init__(self):
        self.ret = self.mtx = self.dist = self.rvecs = self.tvecs = self.focal_length = None
        self.no_photos = 0

    def take_photos(self, no_photos):
        dir_path = 'data/temp'
        basename = 'cc'
        window_name = 'frame'
        ext = 'jpg'
        
        cap = cv2.VideoCapture(CAM_NUMBER)
        if not cap.isOpened():
            return

        os.makedirs(dir_path, exist_ok=True)
        base_path = os.path.join(dir_path, basename)

        n = 0        
        while True:
            if n == no_photos:
                break
            ret, frame = cap.read()
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                print('image saved')
                cv2.imwrite('{}_{}.{}'.format(base_path, n, ext), frame)
                n += 1
            elif key == ord('q'):
                break
            
        cv2.destroyWindow(window_name)

    def calibrate_camera(self, no_photos, no_rows, no_cols):
        try:
            # se defineste criteriul care va opri functia iterativa 'cornerSubPix' cand acest criteriu va fi indeplinit (se refera la eroarea maxima acceptata)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((no_rows*no_cols,3), np.float32)
            objp[:,:2] = np.mgrid[0:no_rows,0:no_cols].T.reshape(-1,2)

            # Arrays to store object points and image points from all the images.
            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.

            images = [r'data/temp/' + 'cc_' +str(i) + '.jpg' for i in range(no_photos)]
            print(len(images))
            if len(images) == 0:
                return
            for fname in images:
                # time.sleep(3)
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                # Find the chess board corners - argumentul al doilea reprezinta numarul de intersectii ale patratelor
                ret, corners = cv2.findChessboardCorners(gray, (no_rows,no_cols), None)

                # If found, add object points, image points (after refining them)
                if ret == True:
                    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (no_rows,no_cols), corners2,ret)
                    cv2.imshow('img',img)
                    k = cv2.waitKey(0)

                    if k & 0xFF == ord('s'):
                        print('skipping')
                        continue
                    
                    objpoints.append(objp)
                    imgpoints.append(corners2)


            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            np.savez('calibrate_camera_parameters.npz', self.mtx, self.dist)
            
            cv2.destroyAllWindows()
        except Exception as e:
            print(e)


def calculate_angle_3_points(a, b, c):
        # radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        radians = np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(a.y - b.y, a.x - b.x)
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        return angle

def calculate_back_angle(a, b):
    c = b
    c.x = b.x + 50
    radians = np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(a.y - b.y, a.x - b.x)
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def angleOf(p1,p2):
    deltaY = p1.y - p2.y
    deltaX = p2.x - p1.x
    result = math.atan2(deltaY, deltaX)
    result = math.degrees(result) + 180

    return result

def calculate_distance_2_points(p1, p2):
        distance = math.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2)
        # import pdb;pdb.set_trace()
        return distance

def calculate_angle_error(default_value, current_value):
        curr_error = abs(current_value - default_value) / default_value
        return curr_error

def main():
    app = DeskMonitoring()
    app.start_application()

    while(True):
        try:
            # Capture the video frame
            ret, frame = app.vid.read()

            app.detectPose(frame, app.pose_image, draw=True, display=True)
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
        except KeyboardInterrupt:
            sys.exit()
        except:
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    app.vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()



if __name__ == '__main__':
    # t1 = threading.Thread(target=calibrate)
    # # starting thread 1
    # t1.start()
    main()