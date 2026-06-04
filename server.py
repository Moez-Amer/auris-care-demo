import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
from model import YamnetClassifier
import json

app = FastAPI()
classifier = YamnetClassifier()

@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to live audio stream.")
    audio_buffer = []
    
    # --- NEW: Streak tracking for loud human sounds ---
    distress_streak = 0
    REQUIRED_STREAK = 3 

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer.extend(chunk)
            
            # Process exactly ~1 second of audio
            if len(audio_buffer) >= 15600:
                input_data = np.array(audio_buffer[:15600], dtype=np.float32)
                
                # --- NEW: The Sliding Window (Shift forward by 0.5s instead of clearing) ---
                audio_buffer = audio_buffer[7800:]
                
                sound_class, confidence = classifier.classify_audio(input_data)
                
                # --- NEW: Enforce the 70% Confidence Threshold ---
                if confidence < 0.70:
                    distress_streak = 0  # Reset streak if confidence drops
                    continue             # Ignore low-confidence noise completely
                
                lower_class = sound_class.lower()
                group = 0
                
                # --- NEW: Reorganized Keyword Categories ---
                # 1. Instant Critical (Falls, Glass, Alarms) -> Trigger immediately
                group_1_instant = ["fire alarm", "smoke detector", "siren", "glass", "shatter", "thump", "thud", "fall"]
                
                # 2. Repeatable Distress (Shouting, Crying) -> Requires streak counter
                group_1_distress = ["shout", "yell", "screaming", "crying", "sobbing", "wail"]
                
                # 3. Monitor (Coughing, Wheezing, Groans) -> Trigger immediately
                group_2_keys = ["cough", "wheeze", "sneeze", "sniff", "throat", "snoring", "breathing", "pant", "vomiting", "moan", "groan", "grunt", "gasp"]
                
                # --- APPLY FILTERING LOGIC ---
                if any(kw in lower_class for kw in group_1_instant):
                    group = 1
                    distress_streak = 0 # Reset streak since we hit a different critical event
                    
                elif any(kw in lower_class for kw in group_1_distress):
                    distress_streak += 1
                    if distress_streak >= REQUIRED_STREAK:
                        group = 1
                        distress_streak = 0 # Reset after sending the alert
                    else:
                        continue # Keep listening silently until streak is met
                        
                elif any(kw in lower_class for kw in group_2_keys):
                    group = 2
                    distress_streak = 0 # Reset streak
                    
                else:
                    distress_streak = 0 # Reset streak for unrelated background noise
                    
                # --- SEND PAYLOAD ---
                if group > 0:
                    response = {
                        "class": sound_class,
                        "confidence": float(confidence), 
                        "group": group
                    }
                    await websocket.send_text(json.dumps(response))
                
    except WebSocketDisconnect:
        print("Client stopped listening and disconnected cleanly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")