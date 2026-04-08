from flask import request
import base64
import numpy as np

@app.route('/detect', methods=['POST'])
def detect():
    global last_status, no_face_frames

    data = request.json['image']

    # remove header
    encoded = data.split(',')[1]
    img_bytes = base64.b64decode(encoded)

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 7, minSize=(50,50))

    if len(faces) == 0:
        no_face_frames += 1
        status = "No Face Detected" if no_face_frames > 10 else "OK"
    else:
        no_face_frames = 0
        status = "Multiple Faces Detected" if len(faces) > 1 else "OK"

    if status != last_status:
        log_event(status)
        last_status = status

    return status