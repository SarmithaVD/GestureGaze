# Import necessary libraries
import os               # Operating system dependent functionality
import cv2              # OpenCV library for computer vision tasks

# Define the directory to store the collected data
DATA_DIR = './data'

# Create the directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes and the size of the dataset
number_of_classes = 30
dataset_size = 100

# Open a connection to the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Iterate through each class
for j in range(number_of_classes):
    # Create a subdirectory for each class
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    # Wait for the user to press 'Q' to start capturing images
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    # Capture and save images until the dataset size is reached
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()