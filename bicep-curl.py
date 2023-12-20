# from flask import Flask, render_template, Response
# from flask_socketio import SocketIO, emit
# from flask_cors import CORS
# import cv2
# import mediapipe as mp
# import numpy as np
# from pathlib import Path

# app = Flask(__name__)
# CORS(app)
# socketio = SocketIO(app)

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# # Curl counter variables for left and right hands
# left_counter = 0
# left_stage = None
# left_prev_stage = None

# right_counter = 0
# right_stage = None
# right_prev_stage = None

# def calculate_angle(a, b, c):
#     a = np.array(a)  # First
#     b = np.array(b)  # Mid
#     c = np.array(c)  # End

#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)

#     if angle > 180.0:
#         angle = 360 - angle

#     return angle

# @socketio.on('connect')
# def handle_connect():
#     print('Client connected')
#     emit('response', {'data': 'Connected'})

# def generate_frames():
#     global left_counter, right_counter, left_stage, left_prev_stage, right_stage, right_prev_stage

#     cap = cv2.VideoCapture(0)

#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()

#             # Recolor image to RGB
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False

#             # Make detection
#             results = pose.process(image)

#             # Recolor back to BGR
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             # Extract landmarks
#             try:
#                 landmarks = results.pose_landmarks.landmark

#                 # Get coordinates for left hand
#                 shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#                 elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#                 wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

#                 # Get coordinates for right hand
#                 shoulder1 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#                 elbow1 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
#                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
#                 wrist1 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

#                 # Calculate angles for left and right hands
#                 angle_left = calculate_angle(shoulder, elbow, wrist)
#                 angle_right = calculate_angle(shoulder1, elbow1, wrist1)

#                 # Visualize angles
#                 cv2.putText(image, f"Left: {angle_left}",
#                             tuple(np.multiply(elbow, [640, 480]).astype(int)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

#                 cv2.putText(image, f"Right: {angle_right}",
#                             tuple(np.multiply(elbow1, [640, 480]).astype(int)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

#                 # Curl counter logic for left hand
#                 if angle_left > 160:
#                     left_stage = "down"
#                 if (angle_left < 30) and left_stage == 'down' and left_prev_stage != 'up':
#                     left_stage = "up"
#                     left_counter += 1
#                     socketio.emit('left_counter', {'count': left_counter})

#                 # Curl counter logic for right hand
#                 if angle_right > 160:
#                     right_stage = "down"
#                 if (angle_right < 30) and right_stage == 'down' and right_prev_stage != 'up':
#                     right_stage = "up"
#                     right_counter += 1
#                     socketio.emit('right_counter', {'count': right_counter})

#                 left_prev_stage = left_stage
#                 right_prev_stage = right_stage

#             except Exception as e:
#                 print(f"Exception: {e}")

#             # Render curl counter
#             # Setup status box
#             cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

#             # Rep data
#             cv2.putText(image, 'RIGHT REPS', (15, 12),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#             cv2.putText(image, str(left_counter),
#                         (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

#             cv2.putText(image, 'LEFT REPS', (15, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#             cv2.putText(image, str(right_counter),
#                         (10, 80),
#                         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

#             # Render detections
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#                                        )

#             ret, buffer = cv2.imencode('.jpg', image)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('bicep-curl-socketio.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# if __name__ == '__main__':
#     socketio.run(app, debug=True)



from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
from pathlib import Path
from threading import Thread
import time
# import eventlet

# eventlet.monkey_patch()
# from gevent import monkey
# monkey.patch_all()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Curl counter variables for left and right hands
left_counter = 0
left_stage = None
left_prev_stage = None

right_counter = 0
right_stage = None
right_prev_stage = None

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('response', {'data': 'Connected'})

def send_frame():
    cap = cv2.VideoCapture(0)
    left_counter = 0
    right_counter = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Check if the frame is empty
            if not ret or frame is None:
                continue
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for left hand
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Get coordinates for right hand
                shoulder1 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow1 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist1 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angles for left and right hands
                angle_left = calculate_angle(shoulder, elbow, wrist)
                angle_right = calculate_angle(shoulder1, elbow1, wrist1)

                # Visualize angles
                cv2.putText(image, f"Left: {angle_left}",
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f"Right: {angle_right}",
                            tuple(np.multiply(elbow1, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Curl counter logic for left hand
                if angle_left > 160:
                    left_stage = "down"
                if (angle_left < 30) and left_stage == 'down' and left_prev_stage != 'up':
                    left_stage = "up"
                    left_counter += 1
                    socketio.emit('left_counter', {'count': left_counter})

                # Curl counter logic for right hand
                if angle_right > 160:
                    right_stage = "down"
                if (angle_right < 30) and right_stage == 'down' and right_prev_stage != 'up':
                    right_stage = "up"
                    right_counter += 1
                    socketio.emit('right_counter', {'count': right_counter})

                left_prev_stage = left_stage
                right_prev_stage = right_stage

            except Exception as e:
                print(f"Exception: {e}")

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'RIGHT REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(left_counter),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'LEFT REPS', (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(right_counter),
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                       )

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            # Emit the frame to the client
            socketio.emit('video_frame', {'frame': base64.b64encode(frame).decode('utf-8')})

def generate_frames():
    # Start a new thread for sending frames
    Thread(target=send_frame).start()

@app.route('/')
def index():
    return render_template('bicep-curl-socketio.html')

@app.route('/video_feed')
def video_feed():
    # Return a response generator
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
