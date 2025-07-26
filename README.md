# 📡 IoT Intrusion Detection System Using Machine Learning

This project focuses on detecting malicious network activities in IoT-enabled Smart Homes by applying Machine Learning (ML) and Deep Learning (DL) algorithms. The system analyzes Wireshark-captured traffic to classify it as **benign** or **malicious**, ensuring better security for resource-constrained IoT devices.

---

## 📁 Dataset

* **Input:** Network traffic captured from Wireshark.
* **File Used:** `IoT_Network_Intrusion.csv` (Top 10,000 rows used).
* **Test File:** `testData.csv`
* **Target:** `label` column (0 = Benign, 1 = Malicious)

---

## 🧰 Technologies & Libraries

* **Language:** Python
* **ML/DL Libraries:**

  * Scikit-learn
  * Keras
  * Seaborn / Matplotlib
* **Feature Selection:** MINE (Maximal Information Coefficient)
* **Preprocessing:** MinMaxScaler, PCA
* **Model Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC

---

## 📊 Models Used

| Algorithm     | Accuracy | Precision | Recall  | F1-Score |
| ------------- | -------- | --------- | ------- | -------- |
| Random Forest | 98.55%   | 98.28%    | 98.77%  | 98.50%   |
| SVM (Sigmoid) | 95.40%   | 95.06%    | 95.50%  | 95.26%   |
| CNN2D         | 100.00%  | 100.00%   | 100.00% | 100.00%  |
| LSTM          | 96.85%   | 96.69%    | 96.83%  | 96.76%   |

---

## 🚀 How It Works

### 1. **Data Loading**

* Loads 10,000 rows from `IoT_Network_Intrusion.csv`.

### 2. **Data Visualization**

* Class label distribution (Benign vs Malicious).

### 3. **Feature Selection**

* Uses **MINE** to select the top 35 informative features.

### 4. **Data Normalization**

* Applies **MinMaxScaler** to normalize selected features.

### 5. **Dimensionality Reduction**

* Applies **PCA** to reduce to 30 principal components.

### 6. **Train/Test Split**

* 80% training, 20% testing.

### 7. **Model Training**

* Trains and evaluates the following:

  * **Random Forest**
  * **Support Vector Machine (SVM)**
  * **CNN2D**
  * **LSTM**

### 8. **Model Evaluation**

* Evaluates all models using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC curves.

### 9. **Test Predictions**

* Reads test data from `testData.csv`, processes it, and predicts using the best-performing model (CNN2D).

---

## 🛠 Requirements

* Python 3.7+
* `pip install` the following:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn keras minepy
```

---

## ✅ Results Summary

* **CNN2D achieved perfect classification (100%) on test data.**
* LSTM also showed strong results, suitable for sequential packet data.
* Traditional ML models (Random Forest, SVM) performed well but not as perfect as CNN2D.

---

## 📌 Conclusion

This project shows the effectiveness of combining ML and DL models for real-time intrusion detection in IoT Smart Homes. Using feature selection, normalization, and deep learning improves accuracy and helps protect low-resource IoT devices from network attacks.

---

