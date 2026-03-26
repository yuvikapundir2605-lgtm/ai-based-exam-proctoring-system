from flask import Flask, Response
import cv2
from datetime import datetime

app = Flask(__name__)

camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def log_event(message):
    with open("log.txt", "a") as f:
        f.write(f"{datetime.now()} - {message}\n")

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            status = "No Face Detected"
            log_event(status)

        elif len(faces) > 1:
            status = "Multiple Faces Detected"
            log_event(status)

        else:
            status = "OK"

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.putText(frame, status, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)