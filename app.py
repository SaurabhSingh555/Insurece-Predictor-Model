from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    features = np.array(data).reshape(1, -1)
    prediction = model.predict(features)
    return render_template('index.html', prediction_text=f'Estimated Insurance Cost: {prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
