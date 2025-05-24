import streamlit as st
import boto3
import json

st.title("🍷 Wine Quality Predictor")
st.markdown("Adjust the sliders below and click **Predict** to estimate wine quality.")

# Input sliders
features = {
    "fixed acidity": st.slider("Fixed Acidity", 4.0, 15.0, 7.4),
    "volatile acidity": st.slider("Volatile Acidity", 0.1, 1.5, 0.7),
    "citric acid": st.slider("Citric Acid", 0.0, 1.0, 0.0),
    "residual sugar": st.slider("Residual Sugar", 0.9, 15.5, 1.9),
    "chlorides": st.slider("Chlorides", 0.01, 0.2, 0.076),
    "free sulfur dioxide": st.slider("Free Sulfur Dioxide", 1, 75, 11),
    "total sulfur dioxide": st.slider("Total Sulfur Dioxide", 6, 289, 34),
    "density": st.slider("Density", 0.9900, 1.0050, 0.9978),
    "pH": st.slider("pH", 2.5, 4.5, 3.51),
    "sulphates": st.slider("Sulphates", 0.2, 2.0, 0.56),
    "alcohol": st.slider("Alcohol", 8.0, 14.9, 9.4)
}

# 🛠️ Replace with your actual SageMaker endpoint name
ENDPOINT_NAME = "your-endpoint-name"  # <-- update this!

if st.button("Predict"):
    try:
        # Prepare input payload
        payload = {"inputs": [list(features.values())]}

        # Call SageMaker endpoint
        client = boto3.client("sagemaker-runtime", region_name="us-east-1")
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )

        result = json.loads(response["Body"].read())
        prediction = result["predictions"][0]

        st.success(f"Predicted Wine Quality: **{prediction}**")
    except Exception as e:
        st.error(f"❌ Prediction failed:\n\n{str(e)}")
