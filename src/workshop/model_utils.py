# Import the necessary packages

# data processing
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Use the logging library
import logging
logger = logging.getLogger(__name__)


def prepare_data(
        data_in: pd.DataFrame,
        selected_features: list,
        y_ID: str,
        test_size: float,
        seed: int = 42) -> tuple:
    """Prepare data for training
    Parameters
    ----------
    data_in:
        pandas table containing attributes as columns,
        observations as row
    selected_features:
        list of strings of features to keep inculding predicting feature
        should match some column names of the data
    y_ID:
        name of the column used as outcome
    test_size:
        fraction of samples to be used as validation data
    seed:
        the random seed to fix the random generators
    Return
    ------
    X_train:
        pandas.DataFrame of the predicting attributes as columns,
        observations as rows
        training subset
    X_test:
        pandas.DataFrame of the predicting attributes as columns,
        observations as rows
        testing subset
    y_train:
        pandas.DataFrame of the outcome attribute as column,
        observations as rows
        training subset
    y_test:
        pandas.DataFrame of the outcome attribute as column,
        observations as rows
        testing subset
    """
    data_subset = data_in[selected_features]
    y = data_subset[y_ID]
    X = data_subset.drop(y_ID, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed)
    return X_train, X_test, y_train, y_test


def plot_evaluation_result(confusion_matrix:np.ndarray) -> None:
    """Plot confusion matrix
    Parameters
    ----------
    confusion_matrix:
        2D array of a confusion matrix of 2 classes (Positive, Negative) 
    """
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_percentages = ["{0:.2%}".format(value) for value in
                        confusion_matrix.flatten()/np.sum(confusion_matrix)]
    labels = [f"{v1}\n{v2}" for v1, v2 in
            zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues')


def train_eval_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """Train model and evaluate the results
    Parameters
    ----------
    model:
        some type of model following sklearn's interface
    X_train:
        training parameters, matrix of shape (n_t, k)
    y_train:
        training labels, matrix of shape (n_t, 1)
    """
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_train)
    cf_matrix = confusion_matrix(y_train.values.ravel(), y_pred)
    plot_evaluation_result(cf_matrix)
    acc = accuracy_score(y_train.values.ravel(), y_pred)
    logger.info(f"The accuracy on the validation data is {acc:.2f}")
    return model


def test_eval_model(model, X_valid: np.ndarray, y_valid: np.ndarray) -> None:
    """Train model and evaluate the results
    Parameters
    ----------
    model:
        some type of model following sklearn's interface
    X_valid:
        training parameters, matrix of shape (n_v, k)
    y_valid:
        training labels, matrix of shape (n_v, 1)
    """
    y_pred = model.predict(X_valid)
    cf_matrix = confusion_matrix(y_valid.values.ravel(), y_pred)
    plot_evaluation_result(cf_matrix)
    acc = accuracy_score(y_valid.values.ravel(), y_pred)
    logger.info(f"The accuracy on the validation data is {acc:.2f}")


def compare_models(
        model_list_names: List,
        model_list: List,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray) -> None:
    """Compare the performance of trained models
    Parameters
    ----------
    model_list_names:
        list of the name of trained models in the model_list
    model_list:
        list of trained models (following sklearns interface)
    X_train:
        training parameters, matrix of shape (n_t, k)
    y_train:
        training labels,  matrix of shape (n_t, 1)
    X_valid:
        training parameters, matrix of shape (n_v, k)
    y_valid:
        training labels, matrix of shape (n_v, 1)
    """
    train_acc = []
    valid_acc = []
    for model in model_list:
        y_t_pred = model.predict(X_train)
        train_acc.append(accuracy_score(y_train.values.ravel(), y_t_pred))
        y_v_pred = model.predict(X_valid)
        valid_acc.append(accuracy_score(y_valid.values.ravel(), y_v_pred))
    
    X_axis = np.arange(len(model_list_names))
    plt.bar(X_axis - 0.2, train_acc, 0.4, label = 'Train')
    plt.bar(X_axis + 0.2, valid_acc, 0.4, label = 'Validation')
    plt.xticks(X_axis, model_list_names)
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Accuracy of trained models on training and validation data")
    plt.legend(loc = 4)
    plt.show()

