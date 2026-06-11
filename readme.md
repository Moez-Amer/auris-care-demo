# Auris SDK: Live Care Monitor Demo

A live demonstration of the **Auris SDK**: a real-time audio intelligence layer for healthcare environments. It captures browser audio, streams it to a Python backend over WebSockets, classifies each 1-second chunk with the **AST (Audio Spectrogram Transformer)** model, maps detections onto a tiered care-home taxonomy, and returns actionable JSON payloads to the UI.

## 🚀 Key Features

* **Tiered Care Taxonomy:** 9 event categories grouped into three response tiers — **CRITICAL** (act now), **URGENT** (prompt check), and **ADVISORY** (logged for clinical review).
* **Exact-Label Matching:** Each category maps to an exact set of AudioSet labels (case-insensitive, no substring matching) to prevent false positives like "Whimper (dog)" firing as human "Crying" or "Church bell" firing as a "Call Bell".
* **Dual-Pane UI:** A live alert feed alongside a real-time SDK JSON payload console.
* **Live Feedback:** A reactive audio volume visualizer and a dark/light theme toggle.

## 🩺 Detection Taxonomy

| Tier | Categories |
|------|------------|
| **CRITICAL** | Fall / Heavy Impact · Choking / Gasping · Scream / Shout · Fire / Smoke Alarm |
| **URGENT** | Crying / Pain · Aggression / Breakage · Medical Alarm / Call Bell |
| **ADVISORY** | Coughing · Sneezing |

## 🛠 Tech Stack

* **AI:** PyTorch + Hugging Face Transformers — `MIT/ast-finetuned-audioset-10-10-0.4593`
* **Backend:** Python, `websockets`, asyncio
* **Frontend:** Vanilla HTML/JS, Web Audio API (`AudioContext` + `getUserMedia`)

---

## ⚙️ Initial Setup

Run these commands once to create the virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch transformers numpy websockets
```

> **First run:** the AST model weights (~350 MB) are downloaded from Hugging Face the first time the server starts, then cached locally.

---

## 💻 How to Run

**1. Start the backend** (loads the AST model and serves the WebSocket on `ws://localhost:8000`):

```bash
source venv/bin/activate
python server.py
```

Wait for `AST Model loaded successfully!` before connecting.

**2. Open the frontend:**

Open `index.html` directly in your browser (double-click it, or drag it into a browser window). Click **Start Listening** and allow microphone access.

> The frontend connects over plain `ws://localhost:8000`. Microphone access and local WebSockets are permitted on `localhost`/`file://` origins without HTTPS, so no certificates are required for the local demo.

## 🔎 What You'll See

* **Terminal:** every detection above the model's 5% floor is logged with its tier, category, specific sound, and confidence.
* **Live Alerts pane:** confident detections rendered as tier-colored alert cards.
* **SDK Output pane:** the JSON payload emitted per event — the raw data your system would pipe into dashboards, records, or notifications.

A detection is forwarded to the UI when its confidence is at or above `UI_CONFIDENCE_FLOOR` (default `0.05` in `server.py`) — tune this for your room and microphone.
