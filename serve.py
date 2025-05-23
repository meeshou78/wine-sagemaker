from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open("wine_model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    features = [input_data[f] for f in input_data]
    prediction = model.predict([features])[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
