import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nn_model import NeuralNetwork

# Define a simple label encoder function
def encode_column(column):
    if column.dtype == object:
        column = column.astype('category').cat.codes
    return column

def build_test_NN(datafile, output_column_name, num_classes, num_features, seed=None, exclude_features=[]):
    # Set the random seed
    # np.random.seed(seed)
    torch.manual_seed(seed)

    # Read the CSV file with headers
    df = pd.read_csv(datafile)

    # Remove unimportant features
    df = df.drop(columns=exclude_features)

    # Encode categorical columns
    df = df.apply(encode_column)

    # Extract features and target variable
    feature_columns = df.columns.difference([output_column_name])
    X = df[feature_columns].to_numpy()
    y = df[output_column_name]

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y.to_numpy()).long()

    y = y.reshape(-1, 1)

    # Convert the target labels to integers
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y)

    y_int = torch.from_numpy(y_int).long()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_int, test_size=0.2, random_state=seed)

    mean = torch.mean(X_train, dim=0)
    std = torch.std(X_train, dim=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std


    # Build a decision tree and train it
    # clf = tree.DecisionTreeClassifier(random_state=seed)
    # clf.fit(X_train, y_train)
    neural_net = NeuralNetwork(num_features, 50, num_classes)
    neural_net = neural_net.fit(X_train, y_train, X_test, y_test, epochs=10, batch_size=32, learning_rate=0.01)

    yh = neural_net.predict(X_test)
    # Test the model on the test set
    acc = accuracy_score(y_test, yh)
    # acc = accuracy_score(y_test, clf.predict(X_test))
    return acc