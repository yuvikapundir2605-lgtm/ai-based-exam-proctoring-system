from flask import Flask, Response, render_template, request
import cv2
from datetime import datetime
import os
import base64
import numpy as np

app = Flask(__name__)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


last_status = ""
no_face_frames = 0


def log_event(message):
    with open("log.txt", "a") as f:
        f.write(f"{datetime.now()} - {message}\n")



@app.route('/')
def home():
    return render_template('index.html')



@app.route('/detect', methods=['POST'])
def detect():
    global last_status, no_face_frames

    data = request.json['image']

    
    encoded = data.split(',')[1]
    img_bytes = base64.b64decode(encoded)

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(50, 50)
    )

    
    if len(faces) == 0:
        no_face_frames += 1
        status = "No Face Detected" if no_face_frames > 3 else "OK"
    else:
        no_face_frames = 0
        if len(faces) > 1:
            status = "Multiple Faces Detected"
        else:
            status = "OK"

    
    if status != last_status:
        log_event(status)
        last_status = status

    return status



@app.route('/status')
def get_status():
    global last_status
    return last_status



@app.route('/tab_switch')
def tab_switch():
    log_event("Tab switched detected")
    return "ok"



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)