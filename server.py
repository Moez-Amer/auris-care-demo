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
    
    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer.extend(chunk)
            
            if len(audio_buffer) >= 15600:
                input_data = np.array(audio_buffer[:15600], dtype=np.float32)
                audio_buffer = audio_buffer[15600:]
                
                sound_class, confidence = classifier.classify_audio(input_data)
                
                # Check for care home keywords
                lower_class = sound_class.lower()
                group = 0
                
                group_1_keys = ["screaming", "crying", "sobbing", "wail", "moan", "groan", "grunt", "gasp", "fire alarm", "smoke detector", "siren", "glass", "shatter", "thump", "thud"]
                group_2_keys = ["cough", "wheeze", "sneeze", "sniff", "throat", "snoring", "breathing", "pant", "vomiting"]
                
                if any(kw in lower_class for kw in group_1_keys):
                    group = 1
                elif any(kw in lower_class for kw in group_2_keys):
                    group = 2
                    
                # Only send data to the phone if it is in Group 1 or Group 2
                if group > 0:
                    response = {
                        "class": sound_class,
                        "confidence": float(confidence), # Send the raw math number to the browser
                        "group": group
                    }
                    await websocket.send_text(json.dumps(response))
                
    except WebSocketDisconnect:
        print("Client stopped listening and disconnected cleanly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")