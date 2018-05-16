import app
import utils
import json
import re
from flask import Flask, request, jsonify
from unittest import TestCase


class AppTests(TestCase):
    def setUp(self):
        app.app.config['TESTING'] = True
        app.app.config['DEBUG'] = True
        self.app = app.app.test_client()

    def test_predict_multiple_models(self):
        app.multiple_versions == True

        response = self.app.post('/predict',
                                 data=json.dumps(dict(id="8db4206f-8878-174d-7a23-dd2c4f4ef5a0",
                                                      score_3=480.0,
                                                      score_4=105.2,
                                                      score_5=0.8514,
                                                      score_6=94.2,
                                                      income=50000)), content_type='application/json')

        json_data = json.loads(response.get_data())

        files = utils.get_files("parquets", "*.parquet")

        assert len(json_data['predictions']) == len(files)

    def test_predict_single_model(self):
        app.multiple_versions = False

        response = self.app.post('/predict',
                                 data=json.dumps(dict(id="8db4206f-8878-174d-7a23-dd2c4f4ef5a0",
                                                      score_3=480.0,
                                                      score_4=105.2,
                                                      score_5=0.8514,
                                                      score_6=94.2,
                                                      income=50000)), content_type='application/json')

        json_data = json.loads(response.get_data())

        assert len(json_data) == 3 and 'predictions' not in json_data

    def test_predict_single_model_is_oldest(self):

        app.multiple_versions = False

        response = self.app.post('/predict',
                                data=json.dumps(dict(id="8db4206f-8878-174d-7a23-dd2c4f4ef5a0",
                                                    score_3=480.0,
                                                    score_4=105.2,
                                                    score_5=0.8514,
                                                    score_6=94.2,
                                                    income=50000)), content_type='application/json')

        json_data = json.loads(response.get_data())

        files = utils.get_files("parquets", "*.parquet")

        dates = [date for f in files for date in re.findall("\d{4}_\d{2}_\d{2}", f)]

        assert app.multiple_versions is False
        assert len(json_data) == 3
        assert dates[0] in json_data['model'] #Check if the model used to make the single prediction corresponds to the oldest model (date)
