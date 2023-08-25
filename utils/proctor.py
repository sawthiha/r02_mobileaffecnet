from typing import List, Dict, Any
from skimage import transform as trans

import cv2

import mediapipe as mp
import numpy as np

from .tflite import *

REFERENCE_LANDMARKS = np.array([
    [38.29459953, 51.69630051], # left eye
    [73.53179932, 51.50139999], # right eye
    [56.02519989, 71.73660278], # nose
    [41.54930115, 92.3655014 ], # left mouth
    [70.72990036, 92.20410156]]) # right mouth

MP_LEFT_IRIS = [
    469, 470, 471, 472
]

MP_RIGHT_IRIS = [
    474, 475, 476, 477
]

MP_FACEMESH_REF_COORS = [
    1, # nose
    61, # Left mouth
    291, # Right mouth
]

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def get_distance(p1, p2):
    return np.sqrt(np.square(p1 - p2).sum(axis=0))


def eval_left_ear(landmarks):
    vert_1 = get_distance(landmarks[158], landmarks[153])
    vert_2 = get_distance(landmarks[160], landmarks[144])
    hor = get_distance(landmarks[33], landmarks[133])
    return (vert_1 + vert_2) / (2 * hor)


def eval_right_ear(landmarks):
    vert_1 = get_distance(landmarks[387], landmarks[373])
    vert_2 = get_distance(landmarks[385], landmarks[380])
    hor = get_distance(landmarks[362], landmarks[263])
    return (vert_1 + vert_2) / (2 * hor)


def eval_left_eld(landmarks):
    return get_distance(landmarks[159], landmarks[145])


def eval_right_eld(landmarks):
    return get_distance(landmarks[386], landmarks[374])


def eval_face_orientation(landmarks):
    return landmarks[1][0], landmarks[1][1]


def standardize(landmarks):
    landmarks = landmarks.landmark
    xs = np.array([landmark.x for landmark in landmarks])
    ys = np.array([landmark.y for landmark in landmarks])
    zs = np.array([landmark.z for landmark in landmarks])
    std_xs = (xs - xs.mean()) / xs.std()
    std_ys = (ys - ys.mean()) / ys.std()
    std_zs = (zs - zs.mean()) / zs.std()
    return np.stack([std_xs, std_ys, std_zs], axis=1)


def landmarks_decor(func):
    def wrapper(landmarks, width, height):
        landmarks = landmarks.landmark
        xs = np.array([landmark.x for landmark in landmarks])
        ys = np.array([landmark.y for landmark in landmarks])
        zs = np.array([landmark.z for landmark in landmarks])
        return func(xs, ys, zs, width, height)

    return wrapper


@landmarks_decor
def candid_ear_model(xs, ys, zs, width, height):
    candid_xs = xs * width
    candid_ys = ys * height
    landmarks = np.stack([candid_xs, candid_ys], axis=1)
    leftEAR = eval_left_ear(landmarks)
    rightEAR = eval_right_ear(landmarks)
    ear = (leftEAR + rightEAR) / 2.0
    return (
        int(ear < 0.2),
        int(leftEAR < 0.2),
        int(rightEAR < 0.2),
        ear,
        leftEAR,
        rightEAR,
    )


@landmarks_decor
def adaptive_eld_model(xs, ys, zs, _, __):
    std_xs = (xs - xs.mean()) / xs.std()
    std_ys = (ys - ys.mean()) / ys.std()
    std_zs = (zs - zs.mean()) / zs.std()
    landmarks = np.stack([std_xs, std_ys], axis=1)
    o_x, o_y = eval_face_orientation(landmarks)

    leftELD = eval_left_eld(landmarks)
    rightELD = eval_right_eld(landmarks)
    eld = (leftELD + rightELD) / 2.0
    threshold = 0.4163 * o_y
    return (
        int(eld < threshold),
        int(leftELD < threshold),
        int(rightELD < threshold),
        eld,
        leftELD,
        rightELD,
    )


@landmarks_decor
def adaptive_ear_model(xs, ys, zs, width, height):
    std_xs = (xs - xs.mean()) / xs.std()
    std_ys = (ys - ys.mean()) / ys.std()
    std_zs = (zs - zs.mean()) / zs.std()
    landmarks = np.stack([std_xs, std_ys], axis=1)
    o_x, o_y = eval_face_orientation(landmarks)

    leftEAR = eval_left_ear(landmarks)
    rightEAR = eval_right_ear(landmarks)
    threshold = (-0.0401 * o_x) + (0.4241 * o_y)
    ear = (leftEAR + rightEAR) / 2.0
    return (
        int(ear < threshold),
        int(leftEAR < threshold),
        int(rightEAR < threshold),
        ear,
        leftEAR,
        rightEAR,
    )


def vertical_alignment(y):
    if y <= 0.058:
        # if y <= -0.05:
        return "up"
    elif y >= 0.6:
        return "down"
    else:
        return "straight"


def horizontal_alignment(x):
    if x <= -0.3:
        return "left"
    elif x >= 0.3:
        return "right"
    else:
        return "straight"

def fixed_image_standardize(image):
    return (image - 127.5) * 0.0078125

def align_face(image, landmarks):
  # Get the left and right eye corner coordinates
  landmarks = (np.array([[landmark.x, landmark.y] for landmark in landmarks.landmark]) * np.array([image.shape[1], image.shape[0]]))
  # compute the center of mass for each eye
  leftEyeCenter = landmarks[MP_LEFT_IRIS].mean(axis=0)
  rightEyeCenter = landmarks[MP_RIGHT_IRIS].mean(axis=0)
  others = landmarks[MP_FACEMESH_REF_COORS]

  src_pts = np.array([leftEyeCenter, rightEyeCenter, *others])

  tform = trans.SimilarityTransform()
  tform.estimate(src_pts, REFERENCE_LANDMARKS)
  tfm = tform.params[0:2, :]

  return cv2.warpAffine(image, tfm, (112, 112))

class MPProctor:

  face_mesh: mp.solutions.face_mesh.FaceMesh
  face_reid: TfLiteWrapper
  face_affect: TfLiteWrapper

  def __init__(self, static_image_mode: bool=True, face_reid_model: str = 'models/face_reid.tflite', face_affect_model: str = 'models/face_affect.tflite'):
    self.face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    self.face_reid = TfLiteWrapper(model_path=face_reid_model)
    self.face_affect = TfLiteWrapper(model_path=face_affect_model)

  def __enter__(self):
      self.face_mesh = self.face_mesh.__enter__()
      return self

  def __exit__(self, exc_type, exc_value, traceback):
      self.face_mesh.__exit__(exc_type, exc_value, traceback)



  __prev_nose: np.ndarray = None
  __prev_landmarks: np.ndarray = None
  def __evaluate(self, image, landmarks) -> Dict[str, Any]:
    cur_nose = np.array(
        [landmarks.landmark[1].x, landmarks.landmark[1].y, landmarks.landmark[1].z]
    )
    delta = (
        np.linalg.norm(cur_nose - self.__prev_nose, ord=2)
        if self.__prev_nose is not None
        else 0.0
    )
    self.__prev_nose = cur_nose

    aligned = align_face(image, landmarks)
    aligned = fixed_image_standardize(aligned)
    inter, embeddings = self.face_reid.predict([aligned])
    expressions, = self.face_affect.predict([inter])

    landmarks = standardize(landmarks)
    o_x, o_y = eval_face_orientation(landmarks)

    leftELD = eval_left_eld(landmarks)
    rightELD = eval_right_eld(landmarks)
    threshold = (
        (-0.0228 * o_x) + (0.0162 * o_y) + (0.0792 * np.exp(np.square(o_y))) + 0.08
    )

    facial_activity = (
        np.linalg.norm(landmarks - self.__prev_landmarks, ord=2)
        if self.__prev_landmarks is not None
        else 0.0
    )
    self.__prev_landmarks = landmarks

    return {
        "horizontal": o_x,
        "horizontal_label": horizontal_alignment(o_x),
        "vertical": o_y,
        "vertical_label": vertical_alignment(o_y),
        "left_eld": leftELD,
        "right_eld": rightELD,
        "threshold": threshold,
        "left_blink": leftELD < threshold,
        "right_blink": rightELD < threshold,
        "facial_activity": facial_activity,
        "movement": delta,
        "embeddings": embeddings,
        "expressions": expressions,
    }

  def process(self, image) -> List[Dict[str, Any]]:
    results = self.face_mesh.process(image)

    if(results.multi_face_landmarks):
      return [
        self.__evaluate(image, landmarks) for landmarks in results.multi_face_landmarks
      ]
    return []