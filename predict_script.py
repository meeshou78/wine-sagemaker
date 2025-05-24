import pickle
import os
import numpy as np

def model_fn(model_dir):
    with open(os.path.join(model_dir, "wine_model.pkl"), "rb") as f:
        return pickle.load(f)

def predict_fn(input_data, model):
    data = np.array([list(input_data.values())])
    return model.predict(data)
