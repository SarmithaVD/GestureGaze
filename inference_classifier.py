# Import necessary libraries
import pickle                               # Module for serializing and deserializing Python objects
import cv2                                  # OpenCV library for computer vision tasks
import mediapipe as mp                       # MediaPipe library for hand landmark detection
import numpy as np                           # NumPy library for numerical operations

# Load the trained model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Open a connection to the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Configure MediaPipe for hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Create a dictionary to map numeric predictions to characters
labels_dict = {i: chr(ord('A') + i) for i in range(26)}

# Continuous loop for real-time hand gesture recognition
while True:
    data_aux = []
    x_ = []
    y_ = []

    # Read a frame from the camera
    ret, frame = cap.read()

    # Get the height, width, and number of channels of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB format for compatibility with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe to detect hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Draw hand landmarks and connections on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Extract hand landmark coordinates
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize and append hand landmark coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Calculate bounding box coordinates
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Make a prediction using the trained model
        prediction = model.predict([np.asarray(data_aux)])

        # Map the numeric prediction to a character
        predicted_character = labels_dict[int(prediction[0])]

        # Draw bounding box and predicted character on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Display the frame with annotations
    cv2.imshow('frame', frame)

    # Wait for a key event, waitKey(1) means wait for 1 millisecond
    cv2.waitKey(1)

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()