# AurisLite

A lightweight, real-time audio classification engine. AurisLite captures live audio from a web browser (desktop or mobile), streams it over WebSockets to a local Python backend, and uses Google's YAMNet machine learning model to instantly classify the sounds being heard.

## Tech Stack
* **AI Model:** TensorFlow / YAMNet
* **Backend:** FastAPI, Uvicorn, WebSockets
* **Frontend:** Vanilla HTML/JS, AudioContext API

---

## How to Run the App

Whenever you want to start the application, open your terminal in this folder and follow these two steps:

### 1. Activate the Environment
You must turn on the isolated Python environment first:
```bash
source venv/bin/activate
```
*(You will know it worked when your terminal prompt starts with `(venv)`)*

### 2. Start the Secure Server
Because modern smartphones block microphone access on plain HTTP connections, we run the server using a local self-signed SSL certificate:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

### 3. Connect!
* **On your Mac:** Open your browser and go to `https://localhost:8000`
* **On your Phone:** Ensure your phone is on the same Wi-Fi as your Mac. Open Safari/Chrome and go to `https://[YOUR-MAC-IP-ADDRESS]:8000`. 
*(Note: You will need to bypass the "Not Secure" warning because it is a self-signed certificate).*

---

## Initial Setup (If installing on a new machine)
If you are cloning this repository to a brand new computer, run these commands to set up the engine:

1. Create the virtual environment: `python3 -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install fastapi uvicorn websockets tensorflow tensorflow-hub numpy`
4. Generate local security keys (for mobile testing): 
   `openssl req -x509 -newkey rsa:2048 -nodes -out cert.pem -keyout key.pem -days 365 -subj "/CN=localhost"`