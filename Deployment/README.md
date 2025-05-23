# 🚀 Deployment - Flask API for Ancient Egyptian Artifact Classifier

This folder contains the deployment setup for the image classification model that identifies ancient Egyptian statues, pyramids, and temples using a trained CNN.

## 🔧 Tech Stack

- Python + Flask (REST API)
- TensorFlow / Keras
- Docker (for containerized deployment)

## 📦 Files

| File | Description |
|------|-------------|
| `app.py` | Flask backend serving the model with a `/predict` endpoint |
| `requirements.txt` | Dependencies for running the API |
| `Dockerfile` | Docker instructions to build and serve the model |
| `brave_pharos_detection_model256.7z` | Compressed Keras model (not included in repo — see below) |

## 🧠 Model

The model is a CNN trained to classify 21 classes of Egyptian artifacts. It was trained on a custom-collected dataset, augmented to ~1000 images per class.

✅ **Model Accuracy:** ~95%  
📁 **Input Shape:** (256, 256, 3)

### 🔗 Download the Model

The model file (`brave_pharos_detection_model256.7z`) can be downloaded from:

👉 [Hugging Face](https://huggingface.co/spaces/monaabdelrazek/AncientAura2/blob/main/brave_pharos_detection_model256.7z)

Replace the `ADD` or `COPY` in `Dockerfile` to use the downloaded file if running locally.

## ▶️ Usage

To run locally via Docker:

```bash
docker build -t egypt-model-api .
docker run -p 7860:7860 egypt-model-api
