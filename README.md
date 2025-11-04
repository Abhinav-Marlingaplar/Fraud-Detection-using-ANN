# Credit Card Fraud Detection using Artificial Neural Network (ANN)

## Project Overview
This project aims to **detect fraudulent credit card transactions** using an **Artificial Neural Network (ANN)** model.  
The model is trained on highly imbalanced transactional data to predict whether a transaction is **legitimate (0)** or **fraudulent (1)** based on several anonymized financial features. The project was developed as part of a **college course requirement** to demonstrate the practical use of ANN models in classification problems.

---

## Dataset
The dataset used is the **Credit Card Fraud Detection dataset** available on Kaggle:  
 [https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)

- **Total Transactions:** 284,807  
- **Fraudulent Transactions:** 492 (≈0.17%)  
- **Type:** Highly imbalanced binary classification dataset  

To avoid local downloads, the dataset was accessed directly from **Google Drive** within Google Colab.

---

## Project Workflow

1. **Data Loading & Preprocessing**
   - Dataset imported from Google Drive  
   - Features standardized using `StandardScaler`  
   - Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset

2. **Model Architecture**
   - Built using Keras Sequential API  
   - Included **Batch Normalization** and **Dropout** layers to prevent overfitting  
   - Tuned hyperparameters using **Keras Tuner**

3. **Model Evaluation**
   - Metrics used: Accuracy, Precision, Recall, F1-score, and ROC-AUC  
   - Evaluation done on both training and testing datasets

4. **Model Saving**
   - Final trained model saved in **Google Drive** in `.keras` format for long-term access

---

## Model Summary

| **Layer Type**              | **Output Shape** | **Parameters** | **Activation Function** | **Purpose / Description** |
|-----------------------------|------------------|----------------|--------------------------|----------------------------|
| **Dense Layer 1**           | (None, 64)       | 1,984          | ReLU                     | Extracts key relationships from the input features. |
| **Batch Normalization 1**   | (None, 64)       | 256            | —                        | Stabilizes activations and speeds up training. |
| **Dropout Layer 1**         | (None, 64)       | 0              | —                        | Randomly deactivates neurons (30%) to prevent overfitting. |
| **Dense Layer 2**           | (None, 16)       | 1,040          | ReLU                     | Learns non-linear patterns from processed data. |
| **Batch Normalization 2**   | (None, 16)       | 64             | —                        | Normalizes activations to improve convergence. |
| **Dropout Layer 2**         | (None, 16)       | 0              | —                        | Reduces overfitting by randomly deactivating neurons (20%). |
| **Dense Layer 3**           | (None, 12)       | 204            | ReLU                     | Final dense layer for refined feature extraction. |
| **Batch Normalization 3**   | (None, 12)       | 48             | —                        | Ensures stable activation distributions before output. |
| **Output Layer**            | (None, 1)        | 13             | Sigmoid                  | Outputs probability of a transaction being fraudulent (1) or legitimate (0). |


---

## Results

**Classification Report:**

           precision    recall  f1-score   support

       0       1.00      1.00      1.00     56864
       1       0.40      0.90      0.55        98

       ROC-AUC Score: 0.947818927796217
       Training Accuracy: 0.9988
       Testing Accuracy: 0.9975



**Inference:**  
The model achieves an exceptionally high **accuracy and ROC-AUC score**, demonstrating its ability to **identify rare fraudulent cases** while maintaining strong generalization performance.

---

## Key Insights
- Applied **SMOTE** improved recall for the minority class (fraudulent transactions).  
- **Batch Normalization + Dropout** stabilized and regularized training.  
- **Hyperparameter tuning** further optimized model performance.  
- The final model successfully minimizes false negatives (missed frauds), which is critical for financial applications.

---

## Technologies Used
- **Python** (Google Colab)
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Scikit-learn** (Preprocessing, Evaluation)
- **TensorFlow / Keras**
- **Imbalanced-learn (SMOTE)**
- **Keras Tuner**
- **Google Drive Integration**

---

## Conclusion
The ANN-based model effectively distinguishes between legitimate and fraudulent transactions with **high recall and ROC-AUC**, making it suitable for deployment in **financial fraud detection systems**.

---

## Author
**Abhinav Marlingaplar**  
B.Tech CSE (AI & DS) | IIIT Kottayam  

