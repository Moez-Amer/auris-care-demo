import asyncio
from fastapi import FastAPI, WebSocket
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
                
                response = {
                    "class": sound_class,
                    "confidence": f"{confidence * 100:.1f}%"
                }
                await websocket.send_text(json.dumps(response))
                
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        await websocket.close()