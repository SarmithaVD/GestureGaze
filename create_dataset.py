# Import necessary libraries
import os               # Operating system dependent functionality
import pickle           # Serialization and deserialization of Python objects
import mediapipe as mp  # MediaPipe library for hand landmark detection
import cv2              # OpenCV library for computer vision tasks

# Configure MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Set the directory path for the dataset
DATA_DIR = './data'

# Initialize lists to store processed data and labels
data = []
labels = []

# Loop through directories in the specified data directory ('./data')
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Temporary list to store data for each image

        x_ = []  # Temporary list to store x-coordinates of landmarks
        y_ = []  # Temporary list to store y-coordinates of landmarks

        # Read the image using OpenCV
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # Convert the image to RGB format using OpenCV
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the RGB image to detect hand landmarks using MediaPipe
        results = hands.process(img_rgb)
        
        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Loop through each landmark and extract x, y coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalize the coordinates and append to data_aux
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Append the processed data and label to the main lists
            data.append(data_aux)
            labels.append(dir_)

# Save the processed data and labels to a pickle file
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()