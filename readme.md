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