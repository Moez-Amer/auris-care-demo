# Auris SDK: Live Care Monitor Demo

A live demonstration of the **Auris SDK**: a real-time audio intelligence layer for healthcare environments. It captures browser audio, streams it to a Python backend via WebSockets, filters for critical events using YAMNet, and returns actionable JSON payloads.

## 🚀 Key Features

* **Targeted Filtering:** Isolates coughs, falls, and alarms while ignoring background noise.
* **Confidence Thresholds:** Requires >70% confidence to prevent false alarms.
* **Dual-Pane UI:** Displays visual alerts alongside a real-time JSON payload console.
* **Live Feedback:** Includes a reactive audio volume visualizer.

## 🛠 Tech Stack

* **AI:** TensorFlow / YAMNet
* **Backend:** Python, FastAPI, Uvicorn, WebSockets
* **Frontend:** Vanilla HTML/JS, AudioContext API
* **Security:** Local SSL

---

## ⚙️ Initial Setup

Run these commands once to configure the environment and generate security keys:

python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn websockets tensorflow tensorflow-hub numpy
openssl req -x509 -newkey rsa:2048 -nodes -out cert.pem -keyout key.pem -days 365 -subj "/CN=localhost"

---

## 💻 How to Run

Start the application from your project folder:

source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000 --ssl-keyfile=key.pem --ssl-certfile=cert.pem

## 📱 Connect

* **On your Mac:** Navigate to https://localhost:8000
* **On your Phone:** Ensure you are on the same Wi-Fi as your Mac. Navigate to https://[YOUR-MAC-IP-ADDRESS]:8000

> **Note:** Because the app uses local self-signed certificates for secure microphone access, you must bypass the "Not Secure" warning in your mobile browser to connect.