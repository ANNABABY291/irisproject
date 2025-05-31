from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model (ensure the model is at 'model/iris_model.pkl')
model = joblib.load('model/iris_model.pkl')

# Mapping from class index to species name
class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input JSON data from the request
    data = request.get_json(force=True)
    
    # Extract the features from the JSON data (sepal length, sepal width, petal length, petal width)
    features = [
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]
    
    # Get the prediction (class index)
    prediction = model.predict([features])
    
    # Map the predicted class index to the species name
    predicted_class_index = prediction[0]
    species_name = class_names.get(predicted_class_index, "Unknown")  # Default to "Unknown" if not found
    
    # Return the species name in the JSON response
    return jsonify({'prediction': species_name})

if __name__ == '__main__':
    # Run the app on all available IPs on port 5000
    app.run(host='0.0.0.0', port=5000)
