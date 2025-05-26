# 🚀 Deployment - Flask API for Ancient Egyptian Artifact Classifier

This folder contains the deployment setup for the image classification model that identifies ancient Egyptian statues, pyramids, and temples using a trained CNN.

## 🔧 Tech Stack

- Python + Flask (REST API)
- TensorFlow / Keras
- Docker (for containerized deployment, optional)
- Hugging Face Spaces (for hosting)

## 📦 Files

| File | Description |
|------|-------------|
| `app.py` | Flask backend serving the model with a `/predict` endpoint |
| `requirements.txt` | Dependencies for running the API |
| `Dockerfile` | Docker instructions to build and serve the model |
| `brave_pharos_detection_model256.7z` | Compressed Keras model (not included in this repo — see link below) |

## 🧠 Model Info

The model is a CNN trained to classify 21 classes of Egyptian artifacts. It was trained on a custom-collected dataset and augmented to ~1000 images per class.

✅ **Model Accuracy:** ~96%  
📁 **Input Shape:** (256, 256, 3)

---

## 🌐 Live API on Hugging Face Spaces

This deployment is already live and running on **Hugging Face Spaces** here:

👉 **[Try the Live API](https://monaabdelrazek-AncientAura2.hf.space/predict)**

> No need to run locally — the API is hosted and served directly from the Hugging Face Space.
> > ⚠️ **Note for Users:**  
> To interact with the live inference API directly, please use a tool like **[Postman](https://www.postman.com/)** or a script (e.g., using Python's `requests` library).  
> Web browsers cannot send file uploads in POST requests correctly, so using Postman is necessary to test the `/predict` endpoint with image files.

---

## 📦 Model File

The model file (`brave_pharos_detection_model256.7z`) used in this deployment is stored on Hugging Face:

🔗 [Download the model from Hugging Face](https://huggingface.co/spaces/monaabdelrazek/AncientAura2/blob/main/brave_pharos_detection_model256.7z)

If you wish to run this project locally, you can download the model and update the `Dockerfile` accordingly.

---

## 🐳 Optional: Run Locally with Docker

> ⚠️ Only needed if you plan to run this API on your own machine.

```bash
# Build Docker image
docker build -t egypt-model-api .

# Run the container
docker run -p 7860:7860 egypt-model-api
