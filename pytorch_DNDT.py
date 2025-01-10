# we have data coming in
# number of classes
# number of features


import torch
from functools import reduce
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def normalize_dataframe_min_max(df):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_normalized


#from paper: https://arxiv.org/pdf/1909.06312.pdf
def torch_kron_prod(a, b):
    res = torch.einsum('ij,ik->ijk', a, b)
    res = res.view(-1, res.shape[1]*res.shape[2])
    return res

def torch_bin(x, cut_points, temperature=0.1):
    D = cut_points.shape[0]
    W = torch.linspace(1.0, D + 1.0, D + 1).view(1, -1)
    cut_points = cut_points.sort()[0]  # make sure cut_points is monotonically increasing
    b = torch.cumsum(torch.cat([torch.tensor([0.0]), -cut_points]), 0)
    h = torch.matmul(x, W) + b
    res = torch.softmax(h / temperature, dim=1)
    return res

def nn_decision_tree(x, cut_points_list, leaf_score, temperature=0.1):
    leaf = reduce(torch_kron_prod, 
                  map(lambda z: torch_bin(x[:, z[0]:z[0] + 1], z[1], temperature), enumerate(cut_points_list)))
    return torch.matmul(leaf, leaf_score)

# Define a simple label encoder function
def encode_column(column):
    if column.dtype == object:
        column = column.astype('category').cat.codes
    return column

#TODO: add seed for torch.manualseed(0)
#TODO: add unimportant features


def build_test_DNDT(datafile, output_column_name, num_features, num_cuts, seed=None, exclude_features=[], normalized=False, test_train_split=False, cuts_per_feat=1):
    if test_train_split:
        return build_test_train_DNDT(datafile, output_column_name, num_features, num_cuts, seed, exclude_features, normalized, cuts_per_feat)
    # Set the random seed for PyTorch
    if seed is not None:
        torch.manual_seed(seed)

    # Read the CSV file with headers
    df = pd.read_csv(datafile)


    # Remove unimportant features
    df = df.drop(columns=exclude_features)

    # Encode categorical columns
    df = df.apply(encode_column)
    if normalized:
        df = normalize_dataframe_min_max(df) #testing idea

    # Extract features and target variable
    feature_columns = df.columns.difference([output_column_name])
    X = df[feature_columns].to_numpy()
    y = df[output_column_name].to_numpy()

    # Convert the species labels to integers
    species_to_int = {species: idx for idx, species in enumerate(np.unique(y))}
    y_int = np.array([species_to_int[eval] for eval in y])

    # Optionally, convert the integer labels to one-hot encoding
    def one_hot_encode(labels, num_classes):
        return np.eye(num_classes)[labels]
    
    num_classes = len(species_to_int)
    y_one_hot = one_hot_encode(y_int, num_classes)

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_one_hot, dtype=torch.float32)
    num_cut = [cuts_per_feat]*num_cuts #num_features  
    num_leaf = np.prod(np.array(num_cut) + 1)
    d = X.shape[1]
    #print(X.shape, y.shape, d, num_cut, num_leaf, num_classes)

    # Initialize variables
    cut_points_list = [torch.nn.Parameter(torch.rand(i)) for i in num_cut]
    leaf_score = torch.nn.Parameter(torch.rand(num_leaf, num_classes))

    # Define loss and optimizer
    optimizer = torch.optim.Adam([*cut_points_list, leaf_score], lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.CrossEntropyLoss()

    # Training loop
    for i in range(1000):
        optimizer.zero_grad()
        y_pred = nn_decision_tree(x_tensor, cut_points_list, leaf_score, temperature=0.1)
        loss = loss_fn(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    with torch.no_grad():
        y_pred_eval = nn_decision_tree(x_tensor, cut_points_list, leaf_score, temperature=0.1)
        error_rate = 1 - (y_pred_eval.argmax(1) == y_tensor.argmax(1)).float().mean()


    y_pred = nn_decision_tree(x_tensor, cut_points_list, leaf_score, temperature=0.1)
    accuracy_score(y_tensor.argmax(1), y_pred.argmax(1))
    return accuracy_score(y_tensor.argmax(1), y_pred.argmax(1))





def build_test_train_DNDT(datafile, output_column_name, num_features, num_cuts, seed=None, exclude_features=[], normalized=False, cuts_per_feat=1):
    
     # Set the random seed for PyTorch
    if seed is not None:
        torch.manual_seed(seed)

    # Read the CSV file with headers
    df = pd.read_csv(datafile)


    # Remove unimportant features
    df = df.drop(columns=exclude_features)

    # Encode categorical columns
    df = df.apply(encode_column)
    if normalized:
        df = normalize_dataframe_min_max(df) #testing idea

    # Extract features and target variable
    feature_columns = df.columns.difference([output_column_name])
    X = df[feature_columns].to_numpy()
    y = df[output_column_name].to_numpy()

    # Convert the species labels to integers
    species_to_int = {species: idx for idx, species in enumerate(np.unique(y))}
    y_int = np.array([species_to_int[eval] for eval in y])

    # Optionally, convert the integer labels to one-hot encoding
    def one_hot_encode(labels, num_classes):
        return np.eye(num_classes)[labels]
    
    num_classes = len(species_to_int)
    y_one_hot = one_hot_encode(y_int, num_classes)


    num_cut = [cuts_per_feat]*num_cuts #num_features  
    num_leaf = np.prod(np.array(num_cut) + 1)
    #d = X.shape[1]
    #print(X.shape, y.shape, d, num_cut, num_leaf, num_classes)

    # Initialize variables
    cut_points_list = [torch.nn.Parameter(torch.rand(i)) for i in num_cut]
    leaf_score = torch.nn.Parameter(torch.rand(num_leaf, num_classes))

    # Define loss and optimizer
    optimizer = torch.optim.Adam([*cut_points_list, leaf_score], lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.CrossEntropyLoss()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_int, test_size=0.2, random_state=seed)
    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)



    # Training loop
    for i in range(1000):
        optimizer.zero_grad()
        y_pred = nn_decision_tree(x_train_tensor, cut_points_list, leaf_score, temperature=0.1)
        loss = loss_fn(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    with torch.no_grad():
        y_pred_test = nn_decision_tree(x_test_tensor, cut_points_list, leaf_score, temperature=0.1)
        accuracy = accuracy_score(y_test_tensor, y_pred_test.argmax(1))

    return accuracy

