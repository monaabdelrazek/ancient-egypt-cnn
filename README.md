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
| `01_Analysis.ipynb`     | Initial EDA, class distribution analysis, imbalance insights |
| `02_Augmentation.ipynb` | Data augmentation techniques to address class imbalance |
| `03_Model_88acc.ipynb`  | First baseline CNN model with 88% accuracy               |
| `04_Model_95acc.ipynb`  | Final refined CNN model with 95% accuracy                |
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

- 📁 **Clean dataset (before augmentation):**  
  [Google Drive Link](PUT-YOUR-GOOGLE-DRIVE-LINK-HERE)

- 📁 **Final dataset (after augmentation & splitting):**  
  [Kaggle Dataset Link](PUT-YOUR-KAGGLE-LINK-HERE)

---

## 🧠 CNN Architecture (Built From Scratch)

A custom CNN architecture was implemented using TensorFlow/Keras (no pre-trained layers). The model was built, tuned, and evaluated iteratively.

### 🔧 Model Highlights:
- 3 Convolutional blocks (Conv2D + ReLU + MaxPooling)
- Dropout layers for regularization
- Fully connected Dense layers
- `Softmax` output layer for multi-class classification
- Custom callbacks for EarlyStopping and ModelCheckpoint

```python


## 📈 Performance

| Metric            | Value                        |
|-------------------|------------------------------|
| **Test Accuracy** | **95%** on unseen data       |
| Loss              | Low & stable                 |
| Overfitting       | Avoided using augmentation & dropout |
| Epochs Trained    | 52 (out of 75 planned epochs) |
| Early Stopping    | **Manual** – Training was stopped at epoch 52 after observing a plateau in validation loss and no further improvement in accuracy |

> 📌 Manual early stopping was used by closely monitoring training and validation performance. Although 75 epochs were planned, training was stopped at epoch 52 when the model stabilized and began to show signs of potential overfitting.


