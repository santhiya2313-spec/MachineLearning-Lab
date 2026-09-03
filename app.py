from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.svm import SVC

app = Flask(__name__)

# Load dataset
df = pd.read_csv("iris_dataset.csv")

X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["target"]

# Train ML model
model = SVC()
model.fit(X, y)


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    features = np.array(data['features']).reshape(1, -1)

    prediction = model.predict(features)

    if prediction[0] == 0:
        result = "Class A"
    elif prediction[0] == 1:
        result = "Class B"
    else:
        result = "Class C"

    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)