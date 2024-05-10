import time
import cv2
import numpy as np
import mediapipe as mp

# Utility functions and classes
class Utils:
    @staticmethod
    def draw_rounded_rect(img, rect_start, rect_end, corner_width, box_color):
        x1, y1 = rect_start
        x2, y2 = rect_end
        w = corner_width

        # draw filled rectangles
        cv2.rectangle(img, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)
        cv2.rectangle(img, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
        cv2.rectangle(img, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
        cv2.rectangle(img, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
        cv2.rectangle(img, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)

        # draw filled ellipses
        cv2.ellipse(img, (x1 + w, y1 + w), (w, w), angle=0, startAngle=-90, endAngle=-180, color=box_color, thickness=-1)
        cv2.ellipse(img, (x2 - w, y1 + w), (w, w), angle=0, startAngle=0, endAngle=-90, color=box_color, thickness=-1)
        cv2.ellipse(img, (x1 + w, y2 - w), (w, w), angle=0, startAngle=90, endAngle=180, color=box_color, thickness=-1)
        cv2.ellipse(img, (x2 - w, y2 - w), (w, w), angle=0, startAngle=0, endAngle=90, color=box_color, thickness=-1)
        return img

    @staticmethod
    def draw_dotted_line(frame, lm_coord, start, end, line_color):
        for i in range(start, end + 1, 8):
            cv2.circle(frame, (lm_coord[0], i), 2, line_color, -1, lineType=cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_text(img, msg, pos, font_scale, font_thickness, text_color, text_color_bg, font=cv2.FONT_HERSHEY_SIMPLEX, width=8, box_offset=(20, 10)):
        offset = box_offset
        x, y = pos
        text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(p - o for p, o in zip(pos, offset))
        rec_end = tuple(m + n - o for m, n, o in zip((x + text_w, y + text_h), offset, (25, 0)))

        img = Utils.draw_rounded_rect(img, rec_start, rec_end, width, text_color_bg)

        cv2.putText(img, msg, (int(rec_start[0] + 6), int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        return text_size

    @staticmethod
    def find_angle(p1, p2, ref_pt=np.array([0, 0])):
        p1_ref = p1 - ref_pt
        p2_ref = p2 - ref_pt
        cos_theta = (np.dot(p1_ref, p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        degree = int(180 / np.pi) * theta
        return int(degree)

    @staticmethod
    def get_landmark_array(pose_landmark, key, frame_width, frame_height):
        denorm_x = int(pose_landmark[key].x * frame_width)
        denorm_y = int(pose_landmark[key].y * frame_height)
        return np.array([denorm_x, denorm_y])

    @staticmethod
    def get_landmark_features(kp_results, dict_features, feature, frame_width, frame_height):
        if feature in ['nose', 'left', 'right']:
            shldr_coord = Utils.get_landmark_array(kp_results, dict_features[feature]['shoulder'], frame_width, frame_height)
            elbow_coord = Utils.get_landmark_array(kp_results, dict_features[feature]['elbow'], frame_width, frame_height)
            wrist_coord = Utils.get_landmark_array(kp_results, dict_features[feature]['wrist'], frame_width, frame_height)
            hip_coord = Utils.get_landmark_array(kp_results, dict_features[feature]['hip'], frame_width, frame_height)
            knee_coord = Utils.get_landmark_array(kp_results, dict_features[feature]['knee'], frame_width, frame_height)
            ankle_coord = Utils.get_landmark_array(kp_results, dict_features[feature]['ankle'], frame_width, frame_height)
            foot_coord = Utils.get_landmark_array(kp_results, dict_features[feature]['foot'], frame_width, frame_height)
            return shldr_coord, elbow_coord, wrist_coord, hip_coord, knee_coord, ankle_coord, foot_coord
        else:
            raise ValueError("feature needs to be either 'nose', 'left' or 'right'")

    @staticmethod
    def get_mediapipe_pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        return mp.solutions.pose.Pose(static_image_mode=static_image_mode, model_complexity=model_complexity, smooth_landmarks=smooth_landmarks, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

# Threshold functions
def get_thresholds_beginner():
    _ANGLE_HIP_KNEE_VERT = {
        'NORMAL': (0, 32),
        'TRANS': (35, 65),
        'PASS': (70, 95)
    }
    return {
        'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,
        'HIP_THRESH': [10, 50],
        'ANKLE_THRESH': 45,
        'KNEE_THRESH': [50, 70, 95],
        'OFFSET_THRESH': 35.0,
        'INACTIVE_THRESH': 15.0,
        'CNT_FRAME_THRESH': 50
    }

def get_thresholds_pro():
    _ANGLE_HIP_KNEE_VERT = {
        'NORMAL': (0, 32),
        'TRANS': (35, 65),
        'PASS': (80, 95)
    }
    return {
        'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,
        'HIP_THRESH': [15, 50],
        'ANKLE_THRESH': 30,
        'KNEE_THRESH': [50, 80, 95],
        'OFFSET_THRESH': 35.0,
        'INACTIVE_THRESH': 15.0,
        'CNT_FRAME_THRESH': 50
    }

# Main processing class
class ProcessFrame:
    def __init__(self, thresholds, flip_frame=False):
        self.flip_frame = flip_frame
        self.thresholds = thresholds
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA
        self.radius = 20
        self.COLORS = {
            'blue': (0, 127, 255),
            'red': (255, 50, 50),
            'green': (0, 255, 127),
            'light_green': (100, 233, 127),
            'yellow': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'cyan': (0, 255, 255),
            'light_blue': (102, 204, 255)
        }
        self.dict_features = {
            'left': {
                'shoulder': 11,
                'elbow': 13,
                'wrist': 15,
                'hip': 23,
                'knee': 25,
                'ankle': 27,
                'foot': 31
            },
            'right': {
                'shoulder': 12,
                'elbow': 14,
                'wrist': 16,
                'hip': 24,
                'knee': 26,
                'ankle': 28,
                'foot': 32
            },
            'nose': 0
        }
        self.state_tracker = {
            'state_seq': [],
            'start_inactive_time': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,
            'DISPLAY_TEXT': np.full((4,), False),
            'COUNT_FRAMES': np.zeros((4,), dtype=np.int64),
            'LOWER_HIPS': False,
            'INCORRECT_POSTURE': False,
            'prev_state': None,
            'curr_state': None,
            'SQUAT_COUNT': 0,
            'IMPROPER_SQUAT': 0
        }
        self.FEEDBACK_ID_MAP = {
            0: ('BEND BACKWARDS', 215, (0, 153, 255)),
            1: ('BEND FORWARD', 215, (0, 153, 255)),
            2: ('KNEE FALLING OVER TOE', 170, (255, 80, 80)),
            3: ('SQUAT TOO DEEP', 125, (255, 80, 80))
        }

    # Add other methods here to complete the functionality
class ProcessFrame:
    # Existing initialization and other methods...

    def _get_state(self, knee_angle):
        """ Determine the current knee state based on angle thresholds. """
        if self.thresholds['HIP_KNEE_VERT']['NORMAL'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['NORMAL'][1]:
            return 's1'
        elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]:
            return 's2'
        elif self.thresholds['HIP_KNEE_VERT']['PASS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['PASS'][1]:
            return 's3'
        return None

    def _update_state_sequence(self, state):
        """ Update the sequence of states based on current state. """
        if state not in self.state_tracker['state_seq']:
            self.state_tracker['state_seq'].append(state)

    def _show_feedback(self, frame, c_frame, dict_maps, lower_hips_disp):
        """ Show feedback on the frame based on current frame analysis. """
        # Implementation for drawing text and other feedback elements on the frame

    def process(self, frame: np.array, pose):
        """ Process a frame for pose analysis and feedback. """
        # Full implementation including extracting keypoints, calculating angles, updating state, and providing feedback.
