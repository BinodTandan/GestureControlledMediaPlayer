import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_frame

# Load the trained model
model = load_model('model/gesture_model.h5')
print("Model loaded successfully!")

IMG_HEIGHT, IMG_WIDTH = 100, 100 

# Gesture mapping
gesture_map = {
    0: 'play',
    1: 'pause',
    2: 'next',
    3: 'previous',
    4: 'stop',
    5: 'volume_up',
    6: 'volume_down',
    7: 'mute'
}

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = preprocess_frame(frame, IMG_HEIGHT, IMG_WIDTH)
    prediction = model.predict(processed_frame)
    predicted_class = np.argmax(prediction)
    action = gesture_map.get(predicted_class, "Unknown")

    # Display the predicted action on the video frame
    cv2.putText(frame, f'Action: {action}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Gesture Control', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
