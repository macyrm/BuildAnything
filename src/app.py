import os
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import uvicorn

MODEL_PATH = os.environ.get("MODEL_PATH", "./models/alzheimer_classifier")
BASE_MODEL = os.environ.get("BASE_MODEL", "distilbert-base-uncased")
API_PORT = int(os.environ.get("API_PORT", 8080))

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

app = Flask(__name__)
CORS(app)

load_model()

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200

@app.route('/screen', methods=['POST'])
def screen_user_response():
    global model, tokenizer
    if model is None or tokenizer is None:
        return jsonify({"error": "Internal Server Error", "message": "Model or tokenizer not loaded. Check server startup logs."}), 500
    try:
        data = request.get_json()
        required_fields = {
            'age': 'Age',
            'mmse_score': 'MMSE Score',
            'memory_complaints': 'Memory Complaints (0/1)',
            'confusion': 'Confusion (0/1)',
            'disorientation': 'Disorientation (0/1)',
            'daily_tasks_difficulty': 'Daily Tasks Difficulty (0/1)'
        }
        extracted_data = {}
        for key, description in required_fields.items():
            value = data.get(key)
            if value is None:
                return jsonify({"error": "Missing field", "message": f"Input JSON must contain a valid value for '{key}' ({description})."}), 400
            extracted_data[key] = value
    except Exception as e:
        return jsonify({"error": f"Invalid input structure or missing field: {e}"}), 400
    text_template = (
        "Patient is a {age}-year-old with an MMSE score of {mmse_score}. "
        "Reported symptoms include memory complaints: {memory_complaints}, "
        "confusion: {confusion}, disorientation: {disorientation}, "
        "and difficulty performing daily tasks: {daily_tasks_difficulty}."
    )
    input_text = text_template.format(**extracted_data)
    try:
        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True)
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.cpu()
        predicted_index = logits.argmax().item()
        label_map = {0: "Normal/Low Risk", 1: "Symptomatic/High Risk"}
        predicted_label = label_map.get(predicted_index, "Unknown")
    except Exception as e:
        return jsonify({"error": "Prediction Failure", "message": f"An error occurred during model inference: {e}"}), 500
    return jsonify({"status": "success", "input_data": extracted_data, "synthesized_text": input_text, "prediction": predicted_label, "prediction_code": predicted_index, "disclaimer": "This result is a project output and is NOT a medical diagnosis."}), 200

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
    # return jsonify({"status": "running", "message": "Welcome to the Alzheimer Classifier API!"})

@app.route('/', methods=['GET'])
def serve_index_html():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    load_model()
    print(f"Starting API server on port {API_PORT}...")
    app.run(host='0.0.0.0', port=API_PORT)