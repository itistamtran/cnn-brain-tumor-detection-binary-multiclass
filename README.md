# 🧠 CNN Brain Tumor Detection (Binary & Multi-Class)

This project uses Convolutional Neural Networks (CNN) to detect brain tumors from MRI scans. It covers **both binary classification** (tumor vs. no tumor) and **multi-class classification** (glioma, meningioma, pituitary, no tumor).

---

## 📁 Project Structure

- `Brain_Tumor_Detection_Binary_Multiclass.ipynb`: Main notebook containing all code for preprocessing, model training, evaluation, and comparison between binary and multiclass CNN models.
- `models/`:
    - Best trained models (.keras files) with clear naming (e.g., binary-model-epoch21-val_acc0.98.keras, multi-model-epoch22-val_acc0.92.keras) are available at the root level for easy loading and inference.
    - The /models/ folder contains additional saved model checkpoints during training for backup and reproducibility purposes.
- `logs/`: TensorBoard logs for training/validation metrics.
- `README.md`: You are here!
  
---

## 📁 Dataset

- **Source**: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
- The dataset includes four categories: `glioma`, `meningioma`, `pituitary`, and `notumor`

---

## 📊 Classification Overview

### Binary Classification:
- **Goal**: Detect whether a tumor is present.
- **Model Accuracy**: ~98%
- **Loss**: ~0.07

### Multi-Class Classification:
- **Goal**: Identify tumor type (glioma, meningioma, pituitary) or no tumor.
- **Model Accuracy**: ~90%
- **Loss**: ~0.48

---

## 📈 Final Model Comparison

| Metric     | Binary Model | Multi-Class Model |
|------------|--------------|-------------------|
| Accuracy   | 0.9773       | 0.9038            |
| F1 Score   | 0.9831       | 0.8873            |
| Loss       | 0.0735       | 0.4822            |

---

## 🧰 Tools & Libraries
- Python 3.x
- TensorFlow / Keras – deep learning framework
- NumPy, Pandas – numerical & data manipulation
- Matplotlib, Seaborn – data visualization
- OpenCV, imutils – image preprocessing
- Pillow (PIL) – image loading
- scikit-learn – metrics, splitting, evaluation
- TensorBoard – model performance visualization
- shutil, zipfile, time, random, os – built-in utilities

## 💻 Development Environment
- Jupyter Notebook
  
---

📌 **Notes**           
EarlyStopping, ModelCheckpoint, and TensorBoard were used to monitor training.

Models are saved as .keras files for easy reloading and inference.

---

💡 **Future Work**                        
Try transfer learning with pre-trained CNNs (ResNet, VGG).

Add Grad-CAM to visualize heatmaps on tumor areas.

Build a simple web app for predictions.

---

👨‍🏫 **Advisor: Dr. Hao Ji**          
**Semester:** Spring 2025  
**Institution:** Cal Poly Pomona

---

Created by **Tam Tran**                    

---

🙏 **Acknowledgments**
  
Dataset from Masoud Nickparvar - Kaggle

Some code structure inspired by [MohamedAliHabib/Brain-Tumor-Detection](https://github.com/MohamedAliHabib/Brain-Tumor-Detection/blob/master/Brain%20Tumor%20Detection.ipynb)
