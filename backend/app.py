from flask import Flask, Response
import cv2
from datetime import datetime

app = Flask(__name__)

camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

last_status = ""
no_face_frames = 0  # ✅ stability fix

def log_event(message):
    with open("log.txt", "a") as f:
        f.write(f"{datetime.now()} - {message}\n")

def generate_frames():
    global last_status, no_face_frames

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ✅ Improved detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(50, 50)
        )

        # 🧠 Stable proctoring logic
        if len(faces) == 0:
            no_face_frames += 1

            if no_face_frames > 10:
                status = "No Face Detected"
            else:
                status = "OK"

        else:
            no_face_frames = 0

            if len(faces) > 1:
                status = "Multiple Faces Detected"
            else:
                status = "OK"

        # ✅ Smart logging
        if status != last_status:
            log_event(status)
            last_status = status

        # Draw boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # Show status
        cv2.putText(frame, status, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        # Convert frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/tab_switch')
def tab_switch():
    log_event("Tab switched detected")
    return "ok"

@app.route('/status')
def get_status():
    global last_status
    return last_status

app.run(debug=True)
