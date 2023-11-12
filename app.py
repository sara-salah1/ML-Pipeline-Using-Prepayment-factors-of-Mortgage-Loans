import pickle
from flask import Flask, render_template, request, jsonify
from myProject1.preprocessing_regression import data_processing_pipeline_regression
from myProject1.feature_addition import create_features
import pandas as pd

flask_app = Flask(__name__)

log_model = pickle.load(open('LogisticRegression.pkl', 'rb'))
reg_model = pickle.load(open('RegressionModel.pkl', 'rb'))

@flask_app.route("/")
def home():
    return render_template("base.html")


@flask_app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data, index=[0])
        prediction = log_model.predict(df)
        print('PREDICTION', prediction)
        prediction_list = prediction.tolist()
        return jsonify({'prediction': prediction_list})

    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)})


@flask_app.route("/predict_regressor", methods=['POST'])
def predict_regressor():
    try:
        data = request.get_json()
        df = pd.DataFrame(data, index=[0])
        df = data_processing_pipeline_regression.fit_transform(df)
        df = create_features(df)
        df = df.drop(['PPR'], axis = 1)
        prediction = reg_model.predict(df)
        print('PREDICTION', prediction)
        prediction_list = (prediction.tolist())*100
        prediction_list = [[prediction_list[0][0]*100]]
        return jsonify({'prediction': prediction_list})

    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    flask_app.run(debug=True)
