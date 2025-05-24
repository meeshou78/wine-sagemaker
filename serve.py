import joblib
import json
import numpy as np
import os

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "wine_model.pkl")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data['inputs'])
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept='application/json'):
    if accept == 'application/json':
        return json.dumps({'predictions': prediction.tolist()}), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

