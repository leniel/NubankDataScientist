import utils
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer as timer
from itertools import repeat
import pandas as pd
from sklearn.linear_model import LogisticRegression
from numpy.random import RandomState
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from pathlib import Path

logger = logging.getLogger('model')

models_path = os.getcwd() + '/models'

def split_dataset(df, validation_percentage, seed):
    state = RandomState(seed)

    validation_indexes = state.choice(df.index, int(
        len(df.index) * validation_percentage), replace=False)

    training_set = df.loc[~df.index.isin(validation_indexes)]

    validation_set = df.loc[df.index.isin(validation_indexes)]

    logger.info("training set has {0} rows".format(len(training_set)))
    logger.info("validation set has {0} rows".format(len(validation_set)))

    return training_set, validation_set


def perform_training(multiple_versions: bool=False):

    parquets = utils.get_files("parquets", "*.parquet")

    utils.delete_create_dir(models_path)

    if multiple_versions:
        with ThreadPoolExecutor(max_workers=10) as executor:
            # launch threads to train the models in parallel
            executor.map(train, parquets)
    else:
        train(parquets[0])


def perform_predictions(post_data, multiple_versions: bool=False):
    """Predicts the probability to send or not a credit card to the customer.

    **Parameters**::
    post_data (json): json post data input.\n
    multiple_versions (bool): specify if prediction is to be performed for multiple model versions.

    **Returns**::
    prediction(s)
    """
    models = utils.get_files("models", "*.pkl")

    if multiple_versions:
        predictions = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            # launch threads to train the models in parallel
            for prediction in executor.map(predict, repeat(post_data), models):
                predictions.append(prediction)

            output = { "id" : post_data['id'], "predictions" : predictions }

            return output
    else:
        prediction = dict(id = post_data['id'])
        
        prediction.update(predict(post_data, models[0]))
        
        return prediction


def train(parquet):
    # load the data
    data = pd.read_parquet(parquet)

    # split into training and validation
    training_set, validation_set = split_dataset(data, 0.25, 1)

    # train model
    training_set["score_3"] = training_set["score_3"].fillna(425)
    training_set["default"] = training_set["default"].fillna(False)
    clf = LogisticRegression(C=0.1)
    clf.fit(training_set[["score_3", "score_4", "score_5",
                          "score_6"]], training_set["default"])

    file_name = Path(parquet).stem + ".pkl"

    # save the trained model to the disk
    joblib.dump(clf, os.path.join(models_path, file_name))


def predict(post_data, model):
    """Predicts the probability to send or not a credit card to the customer.

    **Parameters**::
    post_data: json input
    model: the model from which to extract the prediction
    """

    # loads the model from the disk
    clf = joblib.load(model)

    # transforms the dictionary into a series
    # data gets ordered by key: id, income, score_3, score_4, score_5, score_6
    s = pd.Series(post_data)
    
    # getting only the values...
    x = s.values

    # selecting only the scores (_3 to _6)
    x = x[2:].reshape(1, -1)

    prediction_proba = clf.predict_proba(x)[:, 1]

    prediction = {}
    prediction['model'] = os.path.basename(model)  # the file name only
    prediction['prediction'] = round(prediction_proba[0], 4)

    return prediction


if __name__ == '__main__':
    start = timer()
    perform_training(True)
    end = timer()

    print("{0} seconds".format(end - start))
