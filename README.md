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
- `train_model.ipynb` – CNN model training
- `real_time_control.py` – Real-time gesture recognition and control
- `model/` – Trained Models
- `utils/preprocess.py` – Webcamp frame resizing

## 📥 Dataset:
- Prepare your dataset by organizing gesture images into separate folders for each class under the `data/` directory.
- Each folder should be named according to the gesture label (e.g., `play`, `pause`, `stop`, etc.).

## 🧪 How to Run:

### Clone this repository:
```bash
git clone https://github.com/BinodTandan/GestureControlledMediaPlayer.git
cd gesture-controlled-media-player
```

### Install required dependencies
```bash
pip install -r requirements.txt
```

### Run the training notebook:
Open train_model.ipynb in Jupyter Notebook.
Execute all cells in order.
Ensure that the notebook is fully executed with output cells visible.

### Real-time gesture recognition and media control:
```bash
python real_time_control.py
```
⚠️ Make sure your webcam is connected and the trained model (.h5 file) is located in the model/ directory.

## 🧩 Dependencies:

Python 3.x
TensorFlow / Keras
OpenCV
Numpy
Matplotlib
PyAutoGUI
PyGetWindow

(These can be listed in a requirements.txt file for easy installation.)

## Results Summary:
Baseline CNN: Achieved ~99% accuracy on the test dataset but failed to maintain stable predictions during real-time webcam testing.

MobileNetV2: Achieved 98–100% test accuracy and provided consistent, reliable predictions in real-time recognition scenarios.

Prediction buffer and majority voting significantly improved real-time stability and reduced noisy gesture predictions.


