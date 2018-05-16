## Nubank Data Scientist coding test in Python

#### Description
One of the most important decisions we make here at Nubank is who we give credit cards to. We want to make sure that we're only giving credit cards to people who are likely to pay us back.

We have some data from people who we have given a credit card to in the past, and one of our data scientists has created a model that tries to predict whether someone will pay their credit card bills based on this data.  He claims that the model has really good performance for this problem, with an AUC score of 0.59 (AUC stands for Area Under the receiver operating characteristic Curve, a common performance metric for classification models).

We want to start using this model to make approve or decline decisions in real time, but the data scientist has no idea how to move his research into production.

The data scientist gives you three files:
 - `model.py`: the script which he used to train his model.
 - `training_set.parquet`: the data which he used to train his model.
 - `pip.txt`: the versions of libraries he used in his model

Your task is to create a simple HTTP service that allows us to use this model in production. It should have a POST endpoint `/predict` which accepts and returns JSON content type payloads. Low latency is an important requirement, as other services will hit this endpoint whenever data is available for a new possible customer, and will use the predictions that come from your service to make the decision to send or not a credit card to each customer.

Example input:

```
json
{
    "id": "8db4206f-8878-174d-7a23-dd2c4f4ef5a0",
    "score_3": 480.0,
    "score_4": 105.2,
    "score_5": 0.8514,
    "score_6": 94.2,
    "income": 50000
}
```

Example output:
```
json
{
    "id": "8db4206f-8878-174d-7a23-dd2c4f4ef5a0",
    "prediction": 0.1495
}
```

Once you're comfortable with your solution, you may want to tackle the issue of retraining: Periodically, once we have collected some more data, we may want to retrain the model including this extra data. However, we usually need to keep the old versions working, as they might still be useful. Update your service so that it supports running multiple versions of the model simultaneously. You can assume that when we want to retrain, a parquet file with the new data will be provided.

We will evaluate your code in a similar way that we usually evaluate code that we send to production, so we expect production quality code and tests. Also, pay attention to code organization and make sure it is readable and clean.

You should deliver a git repository with your code and a short README file outlining the solution and explaining how to build and run the code, we just ask that you keep your repository private (GitLab and BitBucket offer free private repositories).

We know this might be your first experience with Python, so don't worry if your code is not idiomatic. Feel free to ask any questions, but please note that we won't be able to give you feedback about your code before your deliver. However, we're more than willing to help you with understanding the domain or picking a library, for instance.

Lastly, there is no need to rush with the solution: delivering your exercise earlier than the due date is not a criteria we take into account when evaluating the exercise: so if you finish earlier than that, please take some time to see what you could improve. Also, if you think the time frame may not be enough for any reason, don't hesitate to ask for more time.

## Solution proposed
### Web API | Data Scientist - Machine Learning Engineer - Nubank
_____________________________________________________________

#### Technologies:
- Programming language: Python 3.6.5 [https://www.python.org/]

- IDE = Visual Studio Code 1.22.1 for Windows [https://code.visualstudio.com/] with

Python extension [https://marketplace.visualstudio.com/items?itemName=ms-python.python].

More info: https://code.visualstudio.com/docs/languages/python

The solution has the following structure:

```
├── app.py (Flask web service endpoint)
├── app.log (app's logging information)
├── description.txt (the problem description)
├── error_handling.py (custom class to handle exceptions)
├── logging.conf (settings used to configure Python logging)
├── model.py (logic to train models and make predictions)
├── pip.txt (project package dependencies)
├── README-txt (this very file)
├── utils.py (useful reusable code)
├── \models (models built)
├── \parquets (parquet file(s) used to train model(s))
├── \tests (unit tests)
```

\* The code has relevant comments where appropriate.

#### Folders
- parquets

Contains the parquet file(s) used to feed the model.

The file name follows this pattern: training_set_yyyy_mm_dd.parquet where:

yyyy = year, mm = month and dd = day

Eg.: training_set_2018_04_07.parquet

When multiple_versions flag in app.py is set to False, the file with the oldest date is picked up by default, that is, the model from which to draw predictions will be trained based on this file.

\* For testing purposes there are 10 parquets in this folder. They're equal.

- models

Trained models are saved to this folder with the help of joblib.dump. This is done to improve reuse.

The file name follows the same pattern as described for parquets. IO is done with pickle [https://docs.python.org/3/library/pickle.html]

- tests

Simple tests to assure the app functionality works as expected.

#### Web Service
The Web service is built with Flask [http://flask.pocoo.org/].

It has a /predict endpoint that accepts only JSON payloads.

Using Visual Studio Code for Windows, to start the web service in Debug mode, select Debug (Ctrl + Shift + D) and Python: Flask (0.11 or later) in the dropdown menu. Click Play to start debugging. You'll see the following command is executed in the Terminal window inside Visual Studio code:

```
PS C:\Repos\NubankDataScientist> cd 'c:\Repos\NubankDataScientist'; ${env:FLASK_APP}='C:\Repos\NubankDataScientist/app.py'; ${env:PYTHONIOENCODING}='UTF-8'; ${env:PYTHONUNBUFFERED}='1'; & 'python' 'C:\Users\leniel\.vscode\extensions\ms-python.python-2018.3.1\pythonFiles\PythonTools\visualstudio_py_launcher.py' 'c:\Repos\NubankDataScientist' '51664' '34806ad9-833a-4524-8cd6-18ca4aa74f14' 'RedirectOutput,RedirectOutput' '-m' 'flask' 'run' '--no-debugger' '--no-reload'
 * Serving Flask app "app"
INFO:werkzeug:  * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit) (2018-04-09 14:11:14; _internal.py:88)
```

Sending requests to the web-service using httpie [https://httpie.org/]:

Open PowerShell and type:

```
PS C:\Users\leniel> http POST localhost:5000/predict id=8db4206f-8878-174d-7a23-dd2c4f4ef5a0, score_3=480.0, score_4=105.2, score_5=0.8514, score_6=94.2, income=50000

HTTP/1.0 200 OK
Content-Length: 954
Content-Type: application/json
Date: Mon, 09 Apr 2018 17:12:32 GMT
Server: Werkzeug/0.14.1 Python/3.6.5

{
    "id": "8db4206f-8878-174d-7a23-dd2c4f4ef5a0",
    "predictions": [
        {
            "model": "training_set_2014_09_10.pkl",
            "prediction": 0.1495
        },
        {
            "model": "training_set_2015_04_06.pkl",
            "prediction": 0.1495
        },
        {
            "model": "training_set_2016_01_10.pkl",
            "prediction": 0.1495
        },
        {
            "model": "training_set_2016_08_13.pkl",
            "prediction": 0.1495
        },
        {
            "model": "training_set_2017_07_17.pkl",
            "prediction": 0.1495
        },
        {
            "model": "training_set_2017_12_17.pkl",
            "prediction": 0.1495
        },
        {
            "model": "training_set_2018_02_22.pkl",
            "prediction": 0.1495
        },
        {
            "model": "training_set_2018_03_01.pkl",
            "prediction": 0.1495
        },
        {
            "model": "training_set_2018_04_05.pkl",
            "prediction": 0.1495
        },
        {
            "model": "training_set_2018_06_06.pkl",
            "prediction": 0.1495
        }
    ]
}
```

The JSON output is returned. multiple_versions is set in app.py and that's why it returned predictions for multiple models. Otherwise only one prediction would've been returned.

#### Logging
Logging information can be seen in the file app.log which stores all relevant information about what's going on with the app including debug info throughout the code, detailed exception stack-trace, etc.

To simulate an exception, in PowerShell type the following command:

```
PS C:\Users\leniel> http --form POST localhost:5000/predict "id=8db4206f-8878-174d-7a23-dd2c4f4ef5a0"
HTTP/1.0 400 BAD REQUEST
Content-Length: 53
Content-Type: application/json
Date: Mon, 09 Apr 2018 17:27:37 GMT
Server: Werkzeug/0.14.1 Python/3.6.5

{
    "message": "predict accepts only JSON payload"
}
```

We tried to send a post request with content-type = form-data and so a brief message explains why no prediction was returned.

To simulate an invalid JSON request, in PowerShell type the following command:

```
PS C:\Users\leniel> http POST localhost:5000/predict id=8db4206f-8878-174d-7a23-dd2c4f4ef5a0, score_3=480.0, score_4=105
.2, score_5=0.8514
HTTP/1.0 500 INTERNAL SERVER ERROR
Content-Length: 157
Content-Type: application/json
Date: Mon, 09 Apr 2018 17:39:52 GMT
Server: Werkzeug/0.14.1 Python/3.6.5

{
    "message": "The following error occurred while processing the request: X has 2 features per sample; expecting 4.\nCheck the log file for more details."
}
```

As can be seen, the output message clearly states what occurred... to see the full stack-trace and get to exactly where in the code the error happened, check the app.log file.
