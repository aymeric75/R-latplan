import pandas as pd
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz, plot_tree
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os 
import pickle
import inspect
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
import joblib
import math
from multiprocessing import Pool, cpu_count, Manager, Value, Lock
from functools import partial
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.problem_transform import ClassifierChain
#from sklearn.multioutput import ClassifierChain
from lightgbm import LGBMClassifier
import lightgbm as lgb
from tqdm import tqdm

import h5py

from itertools import chain


# 


def feature_importance(df, by_best=False):
    # Create a dictionary to store feature scores
    feature_scores = {}

    for col in df.columns:
        # Criterion 1: Fraction of non-missing values (0s and 1s)
        non_missing_count = (df[col] != "?").sum()
        criterion1_score = non_missing_count / len(df)

        # Criterion 2: Entropy of 0s and 1s
        value_counts = df[col].value_counts(normalize=True)
        p0 = value_counts.get(0, 0)  # Proportion of 0s
        p1 = value_counts.get(1, 0)  # Proportion of 1s

        # Calculate entropy (ignore '?' since it's not part of the entropy)
        entropy = -(p0 * np.log2(p0) if p0 > 0 else 0) - (p1 * np.log2(p1) if p1 > 0 else 0)

        # Combine the scores, with the first criterion having higher weight
        overall_score = criterion1_score + 0.5 * entropy

        # Store the score
        feature_scores[col] = overall_score

    if by_best:
        # Sort features by their scores in descending order
        feature_scores = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

    return feature_scores



# Define the function to train the model with feature importance weights
def train_classifier_chain_with_importance(X, Y):
    
    # Calculate feature importance
    importance_scores = feature_importance(X)

    # print("importance_scores")
    # print(importance_scores)
    # exit()

    # Convert importance scores to weights for each feature
    feature_weights = np.array(list(importance_scores.values()))

    # print(feature_weights)
    # exit()
    
    # Normalize the weights
    feature_weights = feature_weights / feature_weights.sum()

    for i, (k, v) in enumerate(importance_scores.items()):
        importance_scores[k] = feature_weights[i]

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    X_train = X_train.replace('?', -1)
    X_test = X_test.replace('?', -1)


    clf = MultiOutputClassifier(DecisionTreeClassifier(criterion='entropy', random_state=42))
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    # Évaluer la performance du modèle (par exemple, par une accuracy moyenne)
    accuracy = np.mean(Y_pred == Y_test)
    print(f"Accuracy: {accuracy}")
    return



    
    



for num_action in range(0, 22):





    #############
    Y = pd.read_csv("Y.txt", sep=" ", header=None)
    Y.name = "Label"
    # Rename columns in Y
    new_y_columns = []
    for i in range(28):
        new_y_columns.append(f"add_{i}")
    for i in range(28):
        new_y_columns.append(f"not(add_{i})")

    Y.columns = new_y_columns[:len(Y.columns)] # Use slicing to handle cases where Y might have fewer columns






    X = pd.read_csv("X.txt", sep=" ", header=None)


    # Rename columns
    new_columns = []
    for i in range(50):
        new_columns.append(f"z_{i}")
    for i in range(50):
        new_columns.append(f"not(z_{i})")

    X.columns = new_columns[:len(X.columns)] # Use slicing to handle cases where X might have fewer columns



    # Create a new DataFrame for the modified X matrix
    new_X = pd.DataFrame()

    # Iterate through the first 50 columns of the original X
    for i in range(50):
        # Create a new column in new_X
        new_column = []
        for index, row in X.iterrows():
            if row[f'z_{i}'] == 1:
                new_column.append(1)
            elif row[f'not(z_{i})'] == 1:
                new_column.append(0)
            else:
                new_column.append('?')  # Or any other representation you prefer for "otherwise"
        new_X[f'z_{i}'] = new_column  # Assign the new column to the new DataFrame

    print("NEW X HEAD")
    print(new_X.head())

    new_X.head(2).to_csv('SEE_new_X_firstTwo.txt', sep='\t', index=False)
    question_marks_count = (new_X.iloc[0] == '?').sum()
    print("question_marks_count first row {}".format(str(question_marks_count)))
    
    print(new_X.shape)



    # importance = feature_importance(new_X)
    # for feature, score in importance:
    #     print(f"Feature {feature}: {score:.4f}")


    # Train the model and get accuracies
    train_classifier_chain_with_importance(new_X, Y)


    exit()
