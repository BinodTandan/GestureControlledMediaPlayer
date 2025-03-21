# Gesture Controlled Media Player 🎮🎵

Control your media player using real-time hand gestures and a CNN-based classifier.

## Current Gestures:
- Play
- Pause
- Volume Up
- Volume Down
- Next Track
- Previous Track
- Mute
- Stop

## How It Works:
- Hand gestures are recognized using a trained CNN model.
- Real-time webcam input is processed.
- Gestures are mapped to media control actions.

## Project Structure:
- `data/` – Gesture images organized by class
- `train_model.py` – CNN model training
- `real_time_control.py` – Real-time gesture recognition and control

