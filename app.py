from flask import Flask, abort, request, jsonify
import model
import logging
import logging.config
import requests
from error_handling import Error

app = Flask(__name__)

multiple_versions = True

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger('app')

@app.errorhandler(Error)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code

    return response

@app.before_first_request
def startup():
    logger.info("Startup started.")
    # train the model(s) the very 1st time a request is made to the web service... subsequent requests won't hit this method.
    model.perform_training(multiple_versions)

    #print("model{0} trained".format("(s)" if multiple_versions else ""))

    logger.info("Startup ended.")


@app.route('/')
def index():
    return 'Web API | Data Scientist - Machine Learning Engineer - Nubank'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.get_json():
            raise Error('predict accepts only JSON payload', status_code=400)

        input_ = request.get_json()

        logger.info('Input = {0}'.format(input_))

        output = model.perform_predictions(request.get_json(), multiple_versions)

        logger.info("Output = {0}".format(output))

        return jsonify(output)
    except Error:
        raise
    except Exception as e:
        logging.fatal(e, exc_info=True) 
        
        raise Error('The following error occurred while processing the request: {0}.\nCheck the log file for more details.'.format(e), status_code=500)
