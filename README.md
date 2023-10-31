OBJECTIVE:  To empower communication for the hearing-impaired by providing a tool that translates sign language gestures into textual representation in real-time.

MODEL USED: RandomForestClassifier

DESCRIPTION:
The system captures hand gestures through a camera, processes them using MediaPipe, and predicts the corresponding sign language character using a trained machine learning model. Utilizing computer vision and machine learning, it detects and interprets hand gestures in real-time, empowering seamless communication for the hearing-impaired.

MODULES:

os: To interact with the operating system, used for file and directory operations.

pickle: Enables the serialization and deserialization of Python objects, employed for saving and loading data.

mediapipe: Offers solutions for various media processing tasks, specifically used for hand landmark detection.

cv2 (OpenCV): A computer vision library used for image and video processing, crucial for capturing and manipulating frames.

sklearn.ensemble: Part of scikit-learn, used for implementing the RandomForestClassifier machine learning model.

sklearn.model_selection: Facilitates data splitting for training and testing the machine learning model.

sklearn.metrics: Includes functions for evaluating the performance of machine learning models, specifically used for accuracy measurement.

numpy: A fundamental package for scientific computing with Python, used for numerical operations and data manipulation.

INSTRUCTIONS:

1) collect_imgs.py:
   
    Create the 'data' directory if it doesn't exist
   
    Define the number of classes and the size of the dataset
   
    Open a connection to the default camera
   
    Iterate through each class, capturing and saving images until the dataset size is reached
   
3) create_dataset.py
   
5) train_classifier.py
   
7) inference_clasifier.py:
   
    Continuous loop for real-time hand gesture recognition
   
    Read a frame from the camera, process with MediaPipe, and predict and display the corresponding sign language character
