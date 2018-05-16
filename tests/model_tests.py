import model
import utils
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from numpy.random import RandomState
from unittest import TestCase


class ModelTests(TestCase):

    def test_split_dataset(self):
        parquets = utils.get_files("parquets", "*.parquet")

        if len(parquets) > 0:
            data = pd.read_parquet(parquets[0])

            # split into training and validation
            training_set, validation_set = model.split_dataset(data, 0.25, 1)

            number_of_customers = len(data)
            customers_to_train = len(training_set)
            customers_to_validate = len(validation_set)

            assert number_of_customers == customers_to_train + customers_to_validate

    def test_predict_multiple_models(self):
        customer = dict(id="8db4206f-8878-174d-7a23-dd2c4f4ef5a0",
                                                    score_3=480.0,
                                                    score_4=105.2,
                                                    score_5=0.8514,
                                                    score_6=94.2,
                                                    income=50000)
        #data = json.dumps(customer)

        predictions = model.perform_predictions(customer, True)

        files = utils.get_files("parquets", "*.parquet")

        assert isinstance(predictions['predictions'], list)
        assert len(predictions['predictions']) == len(files)

    def test_predict_single_model(self):
        customer = dict(id="8db4206f-8878-174d-7a23-dd2c4f4ef5a0",
                                                    score_3=480.0,
                                                    score_4=105.2,
                                                    score_5=0.8514,
                                                    score_6=94.2,
                                                    income=50000)

        prediction = model.perform_predictions(customer, False)

        assert isinstance(prediction, dict)
        assert len(prediction) == 3