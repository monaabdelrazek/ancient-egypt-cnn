
# 🧠 Image Classification from Scratch using CNN

This deep learning project demonstrates the end-to-end development of an image classification system using a **custom Convolutional Neural Network (CNN)** architecture, built entirely **from scratch** — without relying on pre-trained models.

> 📌 **Key Achievements:**  
> ✅ Built CNN model from zero (no transfer learning)  
> ✅ Achieved **96% test accuracy** on unseen data  
> ✅ Designed a robust pipeline including cleaning, augmentation, training, evaluation, and deployment  
> ✅ Deployed on Hugging Face with live API access

---

## 📁 Project Contents

| Notebook                | Description                                              |
|-------------------------|----------------------------------------------------------|
| `01_data_analysis.ipynb`     | Initial EDA, class distribution analysis, imbalance insights |
| `02_data_augmentation_static.ipynb` | Data augmentation techniques to address class imbalance |
| `03_model_attempt1_88acc.ipynb`  | First baseline CNN model with 88% accuracy               |
| `04_model_final_96acc.ipynb`  | Final refined CNN model with 96% accuracy                |
| `class_info.json`| Contains general descriptive information about each class, used to display names during testing and inference |
| `deployment/`           | Scripts and links for online inference API               |
| `example_request.ipynb`       | Upload an image and send it to the deployed Hugging Face API for testing |

> 🗒️ **Note:** The notebooks `01_data_analysis.ipynb` and `02_data_augmentation_static.ipynb` contain some printed messages in Arabic. This **does not affect any code functionality** or the results.

---

## 🧼 Dataset Preparation & Cleaning

The dataset was originally based on publicly available resources, including:

- [Egypt Monuments Dataset on GitHub](https://github.com/marvy-ayman/egypt-monuments-dataset)

Additionally, more images were **manually collected** from various online websites to enrich and balance the dataset.

- Broken/corrupted images were detected and **removed**.
- Validated file formats and color channels (e.g., RGB).
- **Watermarked images were removed** manually.
- Clean replacement images were **manually merged** into the dataset post-cleaning.

> 🧾 **Note:** If the merge step is not visible in the code, it's because it was done offline to ensure full control over dataset quality before training.

---

- After the manual collection and cleaning steps, **data augmentation techniques** were applied locally on the dataset to synthetically increase the number of images per class. This helped to balance the classes and improve model generalization during training.

---

## 🖼️ Dataset Versions

There are two versions available for download:

| Version | Description | Download |
|--------|-------------|----------|
| 🧹 Raw Cleaned Dataset | Contains original cleaned images without augmentation or splitting. Useful if you'd like to redo preprocessing and augmentation from scratch. | [Download from Google Drive](https://drive.google.com/drive/folders/1TedjpusALapU23Aw3k64UG2JnXnpNLMq?usp=sharing) |
| 📦 Final Dataset | Augmented dataset with ~1000 images per class, already split into `train`, `val`, and `test`. Used to train the final model. | [Download from Kaggle](https://www.kaggle.com/datasets/monaabdelrazek/finaldataset/data) |

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
  - The final model, which reached 96% accuracy on the test set.
- Early stopping was applied manually at epoch 52 out of a planned 75 epochs, based on monitoring validation performance.

For detailed implementation and code, please refer to the notebook:
- [`04_model_final_96acc`](https://github.com/monaabdelrazek/ancient-egypt-cnn/blob/main/notebooks/model_final_96acc.ipynb)


---

## 📈 Performance

| Metric            | Value                        |
|-------------------|------------------------------|
| **Test Accuracy** | **96%** on unseen data       |
| Loss              | Low & stable                 |
| Overfitting       | Avoided using augmentation & dropout |
| Epochs Trained    | 52 (out of 75 planned epochs) |
| Early Stopping    | **Manual** – Training was stopped at epoch 52 after observing a plateau in validation loss and no further improvement in accuracy |

> 📌 Manual early stopping was used by closely monitoring training and validation performance. Although 75 epochs were planned, training was stopped at epoch 52 when the model stabilized and began to show signs of potential overfitting.

---
## 🏷️ Supported Classes

The model is trained to classify **exactly 21 distinct ancient Egyptian landmarks**, including royal statues, temples, and pyramids.

Here are the supported classes:

- **King Akhenaten**  
- **King Amenhotep III**  
- **Bent pyramid of Senefru**  
- **Colossi of Memnon**  
- **Goddess Isis**  
- **Queen Hatshepsut**  
- **Khafre Pyramid**  
- **King Thutmose III**  
- **King Tutankhamun**  
- **Queen Nefertiti**  
- **Pyramid of Djoser**  
- **King Ramesses II**  
- **Ramessum (Memorial Temple of Ramesses II)**  
- **King Zoser**  
- **Tutankhamun with Ankhesenamun**  
- **Temple of Hatshepsut**  
- **Temple of Isis in Philae**  
- **Temple of Kom Ombo**  
- **The Great Temple of Ramesses II**  
- **Menkaure Pyramid**  
- **Sphinx**

> ⚠️ The model **cannot predict** labels outside of these 21 predefined classes.

---

## 🚀 Deployment

The final trained model has been deployed and made publicly available for real-time inference.

- 🤖 **Model on Hugging Face:**  
  [🔗 View on Hugging Face](https://huggingface.co/spaces/monaabdelrazek/AncientAura2/blob/main/brave_pharos_detection_model256.7z)

- 🧠 **Live Inference API:**  
  [🔗 Access the API](https://monaabdelrazek-AncientAura2.hf.space/predict)

> ⚠️ **Note for Users:**  
> To interact with the live inference API directly, please use a tool like **[Postman](https://www.postman.com/)** or a script (e.g., using Python's `requests` library).  
> Web browsers cannot send file uploads in POST requests correctly, so using Postman is necessary to test the `/predict` endpoint with image files.


## 🚀 Try it on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/monaabdelrazek/ancient-egypt-cnn/blob/main/example_request.ipynb)

> ⚠️ **Note:** Only `.jpg`, `.jpeg`, and `.png` image formats are supported.  
> Please convert other formats (such as `.heic`) before uploading to avoid errors.



<h3>🖼️ Sample Image for Testing</h3>

<p>To help you try the live API quickly, here's a sample image you can use:</p>

<img src="https://github.com/monaabdelrazek/ancient-egypt-cnn/blob/main/sample_images/King_Zoser.jpeg" alt="King_Zoser" width="300"/>

<p><i>📎 You can right-click the image to <b>save it</b> and use it in the <a href="#-try-it-on-google-colab">Google Colab demo</a> or Postman.</i></p>




