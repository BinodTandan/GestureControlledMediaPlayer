import cv2
import numpy as np
from collections import deque, Counter
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_frame
import pyautogui
import time

# Load the trained model
model = load_model('model/gesture_MobilenetV2_model.h5')
print("Model loaded successfully!")

# Model input shape (same as training)
IMG_HEIGHT, IMG_WIDTH = 224, 224
CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_WINDOW = 5

# Gesture to action mapping (adjust to your needs)
gesture_map = {
    0: 'next',
    1: 'pause',
    2: 'play',
    3: 'previous',
    4: 'stop',
    5: 'volume_up'
}

# Action trigger function using pyautogui
def perform_action(action):
    print(f"Performing action: {action}")
    if action == 'play' or action == 'pause':
        pyautogui.press('space')
    elif action == 'stop':
        pyautogui.press('s')
    elif action == 'next':
        pyautogui.press('n')
    elif action == 'previous':
        pyautogui.press('p')
    elif action == 'volume_up':
        pyautogui.hotkey('ctrl', 'up')
    else:
        print("Unknown action.")

# Prediction buffer and cooldown setup
predictions_buffer = deque(maxlen=SMOOTHING_WINDOW)
last_action = None
cooldown_time = 2  # seconds
last_action_time = time.time()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define ROI
    x, y, w, h = 200, 100, 200, 200
    roi = frame[y:y+h, x:x+w]

    # Preprocess ROI
    processed_frame = preprocess_frame(roi, IMG_HEIGHT, IMG_WIDTH)

    # Model prediction
    prediction = model.predict(processed_frame, verbose=0)
    predicted_prob = np.max(prediction)
    predicted_class = np.argmax(prediction)

    # Add to buffer only if confidence is high
    if predicted_prob >= CONFIDENCE_THRESHOLD:
        predictions_buffer.append(predicted_class)
    else:
        predictions_buffer.append(-1)

    # Apply smoothing (majority vote ignoring -1)
    filtered_preds = [p for p in predictions_buffer if p != -1]
    if filtered_preds:
        final_class = Counter(filtered_preds).most_common(1)[0][0]
        if final_class in gesture_map:
            action = gesture_map[final_class]
            show_action = True
        else:
            action = "Unknown"
            show_action = False
    else:
        action = "Waiting..."
        show_action = False

    # Trigger the action with cooldown logic
    if show_action and action != last_action and time.time() - last_action_time > cooldown_time:
        perform_action(action)
        last_action = action
        last_action_time = time.time()

    # Display action
    if show_action:
        cv2.putText(frame, f'Action: {action}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Confidence: {predicted_prob:.2f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 1)
    else:
        cv2.putText(frame, 'Waiting for clear gesture...', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show ROI box and webcam feed
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Gesture Control', frame)
    cv2.imshow('ROI View', roi)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
