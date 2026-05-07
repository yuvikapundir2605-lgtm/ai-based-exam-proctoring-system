🤖 AI-Based Exam Proctoring System

📌 Description

The AI-Based Exam Proctoring System is a real-time monitoring application designed to ensure fairness during online exams. It uses browser-based webcam access to capture video and sends frames to a Flask backend where OpenCV processes them for face detection. The system identifies suspicious activities such as absence of the candidate, presence of multiple faces, and tab switching. It provides live alerts and maintains a smart log of important events. This project demonstrates the integration of computer vision, web technologies, and cloud deployment.

---

🎯 Features

- 🎥 Browser-based webcam monitoring
- 👤 Face detection using OpenCV
- ❌ No face detection
- 🚨 Multiple faces detection
- 🔄 Tab switching detection
- 🧠 Smart event logging
- 🌐 User-friendly web interface
- ⏱ Exam timer and flow control
- ☁️ Deployed on cloud (Render)

---

🛠 Tech Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python (Flask)
- Computer Vision: OpenCV
- Deployment: Render

---

🚀 How It Works

1. User starts the exam from the web interface
2. Browser requests webcam access
3. Frames are captured and sent to the backend
4. Flask processes frames using OpenCV
5. System detects suspicious activities
6. Alerts are displayed and events are logged

---

⚙️ Installation & Setup

1. Clone Repository

git clone https://github.com/your-username/ai-based-exam-proctoring-system.git
cd ai-based-exam-proctoring-system

2. Install Dependencies

pip install -r requirements.txt

3. Run the Application

python backend/app.py

4. Open in Browser

http://127.0.0.1:5000

---

🌐 Live Demo

👉 https://ai-based-exam-proctoring-system.onrender.com

⚠️ Note: Webcam-based monitoring works best in local environment due to browser security restrictions.


---

📌 Future Enhancements

- Face recognition (identity verification)
- Eye tracking
- Mobile phone detection
- Camera on/off detection
- Admin dashboard

---

👩‍💻 Author

Yuvika Pundir
