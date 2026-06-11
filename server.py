import asyncio
import websockets
import json
import numpy as np
from model import AudioClassifier

# Only events at/above this confidence are shown in the UI. Everything above the
# model's own 5% floor is still printed to the terminal. Tune for your room/mic.
UI_CONFIDENCE_FLOOR = 0.05

print("Loading Heavy AST Model... (This will take a moment)")
classifier = AudioClassifier()
print("AST Model loaded successfully!")

async def handle_audio_stream(websocket):
    print("💻 Local Interface Connected! Listening for audio...")
    
    audio_buffer = np.array([], dtype=np.float32)
    required_samples = 16000 # 1-second chunks

    try:
        async for message in websocket:
            # Convert raw microphone data to numbers the AI understands
            data = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.append(audio_buffer, data)

            if len(audio_buffer) >= required_samples:
                input_data = audio_buffer[:required_samples]
                audio_buffer = audio_buffer[required_samples:]

                # Ask the brain what it heard
                tier, category, specific_sound, confidence = classifier.classify_audio(input_data)

                if category != "Background Noise":
                    # Log EVERY detection to the terminal (with confidence) so we
                    # can see the full picture during testing/demo.
                    print(f"  [{tier:8s}] {category} ({specific_sound}) — {confidence*100:.0f}%")

                    # Only forward confident detections to the UI. Low-confidence
                    # blips (background noise mislabeled at ~5-25%) are real model
                    # outputs but too noisy to show an audience — suppress them.
                    if confidence >= UI_CONFIDENCE_FLOOR:
                        # Map the tier to a UI group: 1=CRITICAL, 2=URGENT, 3=ADVISORY.
                        group_num = {"CRITICAL": 1, "URGENT": 2, "ADVISORY": 3}.get(tier, 3)

                        payload = {
                            "class": f"{category} ({specific_sound})",
                            "category": category,
                            "tier": tier,
                            "confidence": confidence,
                            "group": group_num
                        }

                        await websocket.send(json.dumps(payload))
                    
    except websockets.exceptions.ConnectionClosed:
        print("💻 Interface closed.")
    except Exception as e:
        print(f"An error occurred: {e}")

async def main():
    print("Starting local WebSocket server on ws://localhost:8000...")
    # Bound safely back to localhost!
    async with websockets.serve(handle_audio_stream, "localhost", 8000):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())