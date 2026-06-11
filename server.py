import asyncio
import time
import websockets
import json
import numpy as np
from model import AudioClassifier
from signal_utils import CooldownGate

# --- Demo tuning knobs -------------------------------------------------------
# Only events at/above this confidence reach the UI. The terminal still logs
# everything above the model's lower TERMINAL_DETECTION_FLOOR (see model.py),
# so the two floors are genuinely different. Kept just above the terminal floor
# because AST scores 1 s windows modestly (it expects ~10 s clips); the silence
# gate in model.py — not a high floor — is what suppresses room noise. Raise
# toward 0.25 for fewer/cleaner cards, lower toward 0.10 if real events get cut.
UI_CONFIDENCE_FLOOR = 0.15
# Per-category minimum gap between UI alerts, so a sustained sound (a held
# alarm, a long cry) shows as ONE clean card instead of one per window.
COOLDOWN_SECONDS = 3.0
# Sliding analysis window: classify WINDOW_SAMPLES at a time, advancing by
# HOP_SAMPLES each step. Overlap (50% here) stops an event that straddles a
# hard window boundary from being split into two weak halves and missed.
WINDOW_SAMPLES = 16000   # 1.0 s @ 16 kHz
HOP_SAMPLES = 8000       # advance 0.5 s -> 50% overlap
# If inference can't keep up and audio backs up, drop stale samples so latency
# stays low — better for a live demo than an ever-growing lag.
MAX_BACKLOG_SAMPLES = 48000  # ~3 s

print("Loading Heavy AST Model... (This will take a moment)")
classifier = AudioClassifier()
print("AST Model loaded successfully!")

async def handle_audio_stream(websocket):
    print("💻 Local Interface Connected! Listening for audio...")
    
    audio_buffer = np.array([], dtype=np.float32)
    cooldown = CooldownGate(COOLDOWN_SECONDS)

    try:
        async for message in websocket:
            # Convert raw microphone data to numbers the AI understands
            data = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.append(audio_buffer, data)

            # Slide a window across the buffer with overlap (see HOP_SAMPLES).
            while len(audio_buffer) >= WINDOW_SAMPLES:
                window = audio_buffer[:WINDOW_SAMPLES]
                audio_buffer = audio_buffer[HOP_SAMPLES:]

                # Ask the brain what it heard
                tier, category, specific_sound, confidence = classifier.classify_audio(window)

                if category != "Background Noise":
                    # Log EVERY detection to the terminal (with confidence) so we
                    # can see the full picture during testing/demo.
                    print(f"  [{tier:8s}] {category} ({specific_sound}) — {confidence*100:.0f}%")

                    # Forward to the UI only when it's (a) confident enough AND
                    # (b) not a repeat of a category that just fired. The cooldown
                    # collapses a sustained sound into a single clean alert.
                    if confidence >= UI_CONFIDENCE_FLOOR and cooldown.allow(category, time.time()):
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

            # Keep latency low if inference falls behind realtime: never let the
            # backlog grow without bound — jump to the most recent window.
            if len(audio_buffer) > MAX_BACKLOG_SAMPLES:
                audio_buffer = audio_buffer[-WINDOW_SAMPLES:]

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