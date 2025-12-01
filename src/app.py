import os
from flask import Flask, request, jsonify, send_from_directory, Response
from prometheus_client import Counter, Summary, generate_latest, CONTENT_TYPE_LATEST
from flask_cors import CORS
import joblib
import pandas as pd
import logging

PREDICTIONS_COUNTER = Counter(
    'alz_screener_predictions_total', 
    'Total number of prediction requests received', 
    ['method', 'endpoint']
)

PREDICTION_LATENCY = Summary(
    'alz_screener_prediction_latency_seconds', 
    'Latency of the prediction endpoint in seconds', 
    ['method', 'endpoint']
)

MODEL_PATH = os.environ.get("MODEL_PATH", "./models/xgb_alzheimer_classifier.joblib")
API_PORT = int(os.environ.get("API_PORT", 8080))
FEATURE_MAP = {
    'age': 'Age',
    'mmse_score': 'MMSE',
    'obesity': 'Obesity',
    'medical_history': 'MedicalHistory',
    'cognitive_function': 'CognitiveFunction',
    'symptoms': 'Symptoms'
}
MODEL_FEATURE_COLUMNS = list(FEATURE_MAP.values())
model = None

def load_model():
    global model
    try:
        import os
        if not os.path.exists(MODEL_PATH):
            app.logger.critical(f"Model file not found at expected path: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            
        app.logger.info(f"Attempting to load model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        app.logger.info("Model loaded successfully.")
    except Exception as e:
        app.logger.error(f"Error loading model: {e}") 
        model = None

app = Flask(__name__)
CORS(app) 

app.logger.setLevel(logging.INFO)

load_model()

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/screen', methods=['POST'])
def screen_user_response():
    with PREDICTION_LATENCY.labels(method='POST', endpoint='/screen').time():
        PREDICTIONS_COUNTER.labels(method='POST', endpoint='/screen').inc()
        global model
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        try:
            data = request.get_json()
            extracted_data = {}
            for key in FEATURE_MAP.keys():
                value = data.get(key)
                if value is None:
                    return jsonify({"error": f"Missing field", "message": f"JSON must contain a valid value for '{key}'."}), 400
                extracted_data[key] = value
        except Exception as e:
            return jsonify({"error":f"Invalid input structure or missing field: {e}"}), 400
        
        input_list = [extracted_data[key] for key in FEATURE_MAP.keys()]
        input_df = pd.DataFrame([input_list], columns=MODEL_FEATURE_COLUMNS)
        try:
            predicted_index = model.predict(input_df)[0].item()
            label_map = {0: "Normal/Low Risk", 1: "Symptomatic/High Risk"}
            predicted_label = label_map.get(predicted_index, "Unknown")
        except Exception as e:
            return jsonify({"error": "Prediction Failure", "message": f"Error during model inference: {e}"}), 500
        
        text_template = (
            "Patient is a {age}-year-old with an MMSE score of {mmse_score}. "
            "Reported BMI isi considered obese (BMI >30): {obesity}."
            "Total number of medical history risks (Family History of Alzheimers, Cardiovascular Disease, Diabetes, Depression, Head Injury, Hypertension) reported: {medical_history}."
            "Total number of cognitive function impairments (Behavior Problems, Memory Complaints) reported: {cognitive_function}. "
            "Total number of symptoms (Confusion, Disorientation, Personality Changes, Difficulty Completing Tasks, Forgetfulness) reported: {symptoms}."
        )

        synthesized_text = text_template.format(**extracted_data)

        return jsonify({
            "status": "success",
            "input_data": extracted_data,
            "synthesized_text": synthesized_text,
            "prediction": predicted_label,
            "prediction_code": predicted_index,
            "disclaimer": "This prediction is for informational purposes only and should not be considered a medical diagnosis."
        }), 200

@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/metrics', methods=['GET'])
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    load_model()
    print(f"Starting API server on port {API_PORT}...")