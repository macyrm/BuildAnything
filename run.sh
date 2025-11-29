#!/bin/bash
set -x
IMAGE_NAME="alz-screener-chatbot:latest"
API_PORT=8080
MODEL_PATH="./src/app.py"

echo "--- Starting Script Execution ---"

echo "Cleaning up old artifacts..."
rm -rf ./processed_data
rm -rf ./models
echo "Cleanup complete."

echo "Running data preparation and model training..."
python3 src/data_prep.py
echo "--- STEP 2: data_prep.py finished. Starting train_model.py ---"
python3 src/train_model.py
echo "Model training completed."

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model training failed. Exiting."
  exit 1
fi

echo "Building Docker image..."
docker build -t $IMAGE_NAME .
echo "Docker image built successfully."

echo "Running Docker container..."
docker run --rm -p $API_PORT:$API_PORT --env-file .env.example $IMAGE_NAME
echo "--- Script finished (Container running) ---"