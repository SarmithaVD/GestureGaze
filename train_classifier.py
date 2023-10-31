# Import necessary libraries
import pickle                           # Module for serializing and deserializing Python objects
from sklearn.ensemble import RandomForestClassifier   # Random Forest classifier from scikit-learn
from sklearn.model_selection import train_test_split  # Split arrays into random train and test sets
from sklearn.metrics import accuracy_score             # Accuracy score metric
import numpy as np                       # NumPy library for numerical operations

# Load the preprocessed data from the pickle file
data_dict = pickle.load(open("./data.pickle", 'rb'))

# Convert data and labels to NumPy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check if there is only one unique label or not enough data for training
unique_labels = set(labels)
print(data)
if len(unique_labels) < 2 or len(labels) < 2:
    print("Error: Insufficient unique labels or data for training. Add more samples with different labels.")
else:
    # Perform train-test split only if there are enough unique labels
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    # Initialize a Random Forest Classifier
    model = RandomForestClassifier()

    # Train the model on the training data
    model.fit(x_train, y_train)

    # Predict labels for the test set
    y_predict = model.predict(x_test)

    # Calculate and print the accuracy score
    score = accuracy_score(y_predict, y_test)
    print('{}% of samples were classified correctly!'.format(score * 100))

    # Save the trained model to a pickle file
    f = open('model.p', 'wb')
    pickle.dump({'model': model}, f)
    f.close()