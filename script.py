import cv2
import mediapipe as mp
import numpy as np
import subprocess
import random
import time
import os

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_LIST = ["video1.mp4"] # add your vids here and name them accordingly
FRAME_SCALE = 0.4 # webcam size (smaller = faster but less accurate) 
LOOK_AWAY_TIME = 0.3 # change this to increase/decrease the time before a video plays when looking away
FFPLAY_PATH = r"C:\ffmpeg\bin\ffplay.exe"  # or full path r"C:\ffmpeg\bin\ffplay.exe"

# Eye center tolerance (smaller = stricter)
CENTER_TOL = 0.18

# -----------------------------
# MEDIAPIPE INIT
# -----------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.9 # increase this for better accuracy but worse performance and vice versa
)

# Iris landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eye corners
LEFT_EYE_LEFT = 263
LEFT_EYE_RIGHT = 362
RIGHT_EYE_LEFT = 33
RIGHT_EYE_RIGHT = 133

# -----------------------------
# VIDEO CONTROL
# -----------------------------
video_process = None
last_looking = time.time()

def play_random_video():
    global video_process
    video = random.choice(VIDEO_LIST)
    if not os.path.exists(video):
        print("Missing:", video)
        return
    video_process = subprocess.Popen([
        FFPLAY_PATH,
        "-fs",
        "-autoexit",
        "-loglevel", "quiet",
        video
    ])

def stop_video():
    global video_process
    if video_process and video_process.poll() is None:
        video_process.kill()
    video_process = None

# -----------------------------
# CAMERA
# -----------------------------
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW) # change to 0 if you have only 1 webcam or if 1 doesn't work. You can also try -1 or remove the second argument.
if not cam.isOpened():
    print("Camera failed")
    exit()

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    looking = False

    if result.multi_face_landmarks:
        h, w, _ = frame.shape
        face = result.multi_face_landmarks[0]

        # Get iris center
        left_iris = np.mean([(face.landmark[i].x, face.landmark[i].y) for i in LEFT_IRIS], axis=0)
        right_iris = np.mean([(face.landmark[i].x, face.landmark[i].y) for i in RIGHT_IRIS], axis=0)

        # Get eye corners
        l_left = face.landmark[LEFT_EYE_LEFT]
        l_right = face.landmark[LEFT_EYE_RIGHT]
        r_left = face.landmark[RIGHT_EYE_LEFT]
        r_right = face.landmark[RIGHT_EYE_RIGHT]

        # Horizontal gaze ratio
        l_ratio = (left_iris[0] - l_left.x) / (l_right.x - l_left.x + 1e-6)
        r_ratio = (right_iris[0] - r_left.x) / (r_right.x - r_left.x + 1e-6)

        gaze = (l_ratio + r_ratio) / 2

        if 0.5 - CENTER_TOL < gaze < 0.5 + CENTER_TOL:
            looking = True

    # -------------------------
    # LOGIC
    # -------------------------
    if looking:
        last_looking = time.time()
        stop_video()
    else:
        if time.time() - last_looking > LOOK_AWAY_TIME:
            if video_process is None:
                play_random_video()

    cv2.imshow("Camera (MediaPipe)", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
stop_video()
