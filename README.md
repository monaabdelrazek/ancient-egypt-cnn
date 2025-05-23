# ancient-egypt-cnn
# 🧠 Image Classification from Scratch using CNN

This deep learning project demonstrates the end-to-end development of an image classification system using a **custom Convolutional Neural Network (CNN)** architecture, built entirely **from scratch** — without relying on pre-trained models.

> 📌 **Key Achievements:**  
> ✅ Built CNN model from zero (no transfer learning)  
> ✅ Achieved **95% test accuracy** on unseen data  
> ✅ Designed a robust pipeline including cleaning, augmentation, training, evaluation, and deployment  
> ✅ Deployed on Hugging Face with live API access

---

## 📁 Project Contents

| Notebook                | Description                                              |
|-------------------------|----------------------------------------------------------|
| `01_data_analysis.ipynb`     | Initial EDA, class distribution analysis, imbalance insights |
| `02_data_augmentation_static.ipynb` | Data augmentation techniques to address class imbalance |
| `03_model_attempt1_88acc.ipynb`  | First baseline CNN model with 88% accuracy               |
| `04_model_final_95acc.ipynb`  | Final refined CNN model with 95% accuracy                |
| `class_info.json`| Contains general descriptive information about each class, used to display names during testing and inference |
| `deployment/`           | Scripts and links for online inference API               |

---

## 🧼 Dataset Preparation & Cleaning

The dataset was **manually collected** from online sources and processed through an extensive cleaning pipeline:

- Broken/corrupted images were detected and **removed**.
- Validated file formats and color channels (e.g., RGB).
- **Watermarked images were removed** manually.
- Clean replacement images were **manually merged** into the dataset post-cleaning.

> 🧾 **Note:** If the merge step is not visible in the code, it's because it was done offline to ensure full control over dataset quality before training.

---

## 🖼️ Dataset Versions

There are two versions available for download:

| Version | Description | Download |
|--------|-------------|----------|
| 🧹 Raw Cleaned Dataset | Contains original cleaned images without augmentation or splitting. Useful if you'd like to redo preprocessing and augmentation from scratch. | [Download from Google Drive](https://drive.google.com/drive/folders/1TedjpusALapU23Aw3k64UG2JnXnpNLMq?usp=drive_link) |
| 📦 Final Dataset | Augmented dataset with ~1000 images per class, already split into `train`, `val`, and `test`. Used to train the final model. | [Download from Kaggle](https://www.kaggle.com/datasets/monaabdelrazek/finaldataset) |

⚠️ These datasets are used for educational purposes only and were collected from publicly available sources.

---

## 🧠 CNN Model Architecture Overview

The deep learning model is a custom-built Convolutional Neural Network (CNN) designed from scratch, without using any pre-trained weights.

### Architecture Summary:
- The network starts with a convolutional layer of 32 filters, followed by progressively deeper convolutional layers with 64, 128, and 256 filters.
- After some of the convolutional layers, MaxPooling is applied to reduce spatial dimensions.
- The feature maps are then flattened and passed through a fully connected dense layer with 512 neurons, followed by a dropout layer (rate 0.1) to reduce overfitting.
- The final output layer uses a softmax activation with 21 units, corresponding to the 21 classes in the dataset.

### Training Details:
- Two models were trained during experimentation:
  - An initial model achieving 88% accuracy.
  - The final model, which reached 95% accuracy on the test set.
- Early stopping was applied manually at epoch 52 out of a planned 75 epochs, based on monitoring validation performance.

For detailed implementation and code, please refer to the notebook:
- [`04_model_final_95acc`](https://github.com/monaabdelrazek/ancient-egypt-cnn/blob/main/notebooks/model_final_95acc.ipynb)


---

## 📈 Performance

| Metric            | Value                        |
|-------------------|------------------------------|
| **Test Accuracy** | **95%** on unseen data       |
| Loss              | Low & stable                 |
| Overfitting       | Avoided using augmentation & dropout |
| Epochs Trained    | 52 (out of 75 planned epochs) |
| Early Stopping    | **Manual** – Training was stopped at epoch 52 after observing a plateau in validation loss and no further improvement in accuracy |

> 📌 Manual early stopping was used by closely monitoring training and validation performance. Although 75 epochs were planned, training was stopped at epoch 52 when the model stabilized and began to show signs of potential overfitting.

---

## 🚀 Deployment

The final trained model has been deployed and made publicly available for real-time inference.

- 🤖 **Model on Hugging Face:**  
  [🔗 View on Hugging Face](https://huggingface.co/spaces/monaabdelrazek/AncientAura2/blob/main/brave_pharos_detection_model256.7z)

- 🧠 **Live Inference API:**  
  [🔗 Access the API](https://monaabdelrazek-AncientAura2.hf.space/predict)

### 🔍 Example Usage (Python):
```python
import requests

url = "[(https://huggingface.co/spaces/monaabdelrazek/AncientAura2/blob/main/brave_pharos_detection_model256.7z)"
files = {"file": open("Queen_Hatshepsut.jpg", "rb")}
response = requests.post(url, files=files)

print(response.json())
