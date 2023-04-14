import numpy as np
import tensorflow as tf

def load_data():
    """
    Loads the test data and labels.

    Returns:
    test_data (numpy.ndarray): The test data.
    test_labels (numpy.ndarray): The test labels.
    """
    test_data = np.load('test_data.npy')
    test_labels = np.load('test_labels.npy')
    return test_data, test_labels

def load_models():
    """
    Loads the saved models.

    Returns:
    models (list): The loaded models.
    """
    model_paths = ['voter1.h5', 'voter2.h5', 'voter3.h5', 'voter4.h5', 'voter5.h5']
    models = []
    for model_path in model_paths:
        models.append(tf.keras.models.load_model(model_path))
    return models

def get_predictions(models, test_data):
    """
    Gets the predictions for each model on the test data.

    Args:
    models (list): The models.
    test_data (numpy.ndarray): The test data.

    Returns:
    predictions (numpy.ndarray): The predictions for each model.
    """
    predictions = []
    for model in models:
        predictions.append(np.argmax(model.predict(test_data), axis=1))
    predictions = np.array(predictions)
    return predictions

def majority_vote(predictions):
    """
    Calculates the majority vote for each data point.

    Args:
    predictions (numpy.ndarray): The predictions for each model.

    Returns:
    majority_vote (numpy.ndarray): The majority vote for each data point.
    """
    majority_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
    return majority_vote

def calculate_accuracy(majority_vote, test_labels):
    """
    Calculates the accuracy of the majority voting.

    Args:
    majority_vote (numpy.ndarray): The majority vote for each data point.
    test_labels (numpy.ndarray): The test labels.

    Returns:
    accuracy (float): The accuracy of the majority voting.
    """
    accuracy = np.mean(majority_vote == test_labels)
    return accuracy

def calculate_reliable_votes(majority_vote, n):
    """
    Calculates the required number of votes for a reliable decision.

    Args:
    majority_vote (numpy.ndarray): The majority vote for each data point.
    n (int): The number of models.

    Returns:
    reliable_votes (float): The required number of votes for a reliable decision.
    """
    k = np.sum(majority_vote == test_labels)
    reliable_votes = np.ceil((n+1)/2)
    return reliable_votes

def calculate_probability_correct_decision(majority_vote, n, reliable_votes):
    """
    Calculates the probability of a correct decision using Condorcet's Jury Theorem.

    Args:
    majority_vote (numpy.ndarray): The majority vote for each data point.
    n (int): The number of models.
    reliable_votes (float): The required number of votes for a reliable decision.

    Returns:
    p_correct (float): The probability of a correct decision.
    """
    k = np.sum(majority_vote == test_labels)
    p_correct = np.sum(np.arange(k, reliable_votes+1) / n)
    return p_correct

if __name__ == '__main__':
    # Load the data
    test_data, test_labels = load_data()

    # Load the models
    models = load_models()

    # Get the predictions for each model
    predictions = get_predictions(models, test_data)

    # Calculate the majority vote for each data point
    majority_vote = majority_vote(predictions)

    # Calculate the accuracy of the majority voting
    accuracy = calculate_accuracy(majority_vote, test_labels)
    print('Accuracy:', accuracy)

    # Calculate the required number of votes for a reliable decision
    reliable_votes = calculate_reliable_votes(majority_vote, len(models))
    print('Reliable votes:', reliable_votes)

    # Calculate the probability of a correct decision using Condorcet's Jury Theorem
    p_correct = calculate_probability_correct_decision(majority_vote, len(models), reliable_votes)
    print('Probability of correct decision:', p_correct)