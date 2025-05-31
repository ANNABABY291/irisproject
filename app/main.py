from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model/iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Extract features from the JSON keys (adjust if needed)
    features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
    prediction = model.predict([features])
    predicted_class = prediction[0]
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


