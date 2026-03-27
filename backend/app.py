from flask import Flask, Response
import cv2
from datetime import datetime

app = Flask(__name__)

# Start camera
camera = cv2.VideoCapture(0)

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ✅ Global variable (important)
last_status = ""

# Logging function
def log_event(message):
    with open("log.txt", "a") as f:
        f.write(f"{datetime.now()} - {message}\n")

def generate_frames():
    global last_status   # ✅ use global

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # 🧠 Proctoring logic
        if len(faces) == 0:
            status = "No Face Detected"

        elif len(faces) > 1:
            status = "Multiple Faces Detected"

        else:
            status = "OK"

        # ✅ Smart logging (only when status changes)
        if status != last_status:
            log_event(status)
            last_status = status

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Show status on screen
        cv2.putText(frame, status, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        # Convert frame for browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream to browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)