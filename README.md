# 🧠 EEG-Based Seizure Prediction & Classification

<p align="center">
  <b>⚡ Machine Learning for EEG Signal Analysis</b><br>
  <i>Feature Engineering • Signal Preprocessing • Regularization • Class Imbalance • Logistic Regression</i>
</p>

---

## 🧠 Project Overview

This project focuses on **EEG signal analysis and seizure classification using Machine Learning**.

The notebook investigates how different **data preprocessing, feature extraction, dimensionality reduction, regularization, and class-imbalance techniques** affect the performance of an EEG classification model.

The primary classification algorithm used in the project is **Logistic Regression**.

The experiments are designed to determine which combination of preprocessing and modeling techniques provides the best classification performance for EEG-based seizure prediction.

---

## 🎯 Project Objectives

**The main objectives are:**

* 🧠 Analyze EEG signal data
* 🧹 Perform data preprocessing
* 📏 Standardize EEG features
* 🔇 Apply signal smoothing/denoising
* 🔍 Select informative features
* 📉 Investigate dimensionality reduction
* 🤖 Build a Logistic Regression classifier
* ⚖️ Study the effect of regularization
* 📊 Compare L1, L2, and Elastic Net
* ⚠️ Investigate class imbalance
* 🔄 Compare SMOTE, undersampling, and class weighting
* 📈 Evaluate models using F1-score and PR-AUC
* 🏆 Identify the most effective preprocessing and modeling strategy

---

# 📂 Datasets

**The notebook loads three EEG-related datasets:**

### 1. EEG Feature Dataset

**Used for exploring EEG features and their statistical properties.**

### 2. Epileptic Seizure Recognition Dataset

**Used as an EEG/seizure-related dataset for the project experiments.**

**The target variable `y` is converted into a binary classification format:**

```text
1 → Seizure
0 → Non-Seizure
```

### 3. EEG Eye State Dataset

**The notebook uses the `eyeDetection` variable as the target for the main classification pipeline.**

**The feature matrix is created by removing the target column:**

```python
X = df3.drop(columns=["eyeDetection"])
y = df3["eyeDetection"]
```

---

# 🔄 Machine Learning Workflow

```text
                 🧠 EEG DATA
                     │
                     ▼
              🧹 DATA CLEANING
                     │
                     ▼
              📏 STANDARDIZATION
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    🔵 PIPELINE A          🟣 PIPELINE B
          │                     │
     Denoising            Statistical Features
          │                     │
   Variance Selection           │
          │                     ▼
          │                    PCA
          │                     │
          └──────────┬──────────┘
                     ▼
              🤖 LOGISTIC
              REGRESSION
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        L1/L2    Elastic Net   Class Balance
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
                  SMOTE   Undersampling  Weights
                     │
                     ▼
                📊 EVALUATION
                     │
             F1 Score / PR-AUC
```

---

# 🧹 Data Preprocessing

**The project applies several preprocessing techniques to prepare the EEG data.**

### Missing Value Handling

**Missing values are replaced using the mean of the corresponding feature:**

```python
X = X.fillna(X.mean())
```

**This prevents missing values from interfering with the machine-learning algorithms.**

---

# 📏 Feature Standardization

**The EEG features are standardized using `StandardScaler`.**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Standardization transforms the features to a comparable scale, which is particularly important for Logistic Regression and PCA.**

---

# 🔇 EEG Signal Denoising

**The project applies a rolling-window smoothing technique to the standardized EEG features.**

```python
X_denoised = X_scaled_df.rolling(
    window=3,
    min_periods=1
).mean()
```

### Purpose

**The smoothing operation is intended to reduce short-term fluctuations and noise in the EEG feature representation.**

---

# 🔍 Pipeline A — Feature Selection

**Pipeline A focuses on preserving the original EEG feature structure while removing features with extremely low variance.**

The project uses:

```python
VarianceThreshold(threshold=0.01)
```

### Pipeline A

```text
Raw EEG Features
       ↓
Standardization
       ↓
Smoothing / Denoising
       ↓
Variance Threshold
       ↓
Selected Features
       ↓
Logistic Regression
```

**This approach retains substantially more of the original EEG information.**

---

# 🧮 Pipeline B — Statistical Feature Extraction + PCA

**Pipeline B performs aggressive feature compression.**

**For each EEG observation, statistical features are extracted:**

* Mean
* Standard deviation
* Maximum
* Minimum
* Range

```python
X_features = pd.DataFrame({
    "mean": X_df.mean(axis=1),
    "std": X_df.std(axis=1),
    "max": X_df.max(axis=1),
    "min": X_df.min(axis=1),
    "range": X_df.max(axis=1)
})
```

These features are then standardized and passed through **Principal Component Analysis (PCA)** while retaining 95% of the variance.

### Pipeline B

```text
Raw EEG Features
       ↓
Statistical Feature Extraction
       ↓
Mean / Std / Max / Min / Range
       ↓
Standardization
       ↓
PCA (95% Variance)
       ↓
Logistic Regression
```

---

# 📊 Pipeline Comparison

| Pipeline      | Approach                       | F1 Score |   PR-AUC |
| ------------- | ------------------------------ | -------: | -------: |
| 🔵 Pipeline A | Denoising + Variance Selection | **0.43** | **0.57** |
| 🟣 Pipeline B | Statistical Features + PCA     |     0.21 |     0.48 |

**According to the notebook, Pipeline A substantially outperformed Pipeline B. The interpretation given is that Pipeline B compressed the rich EEG representation too aggressively, resulting in information loss.**

---

# 🤖 Classification Model

**The primary machine-learning algorithm used is:**

## Logistic Regression

**Logistic Regression is used as the baseline classifier and throughout the regularization and class-imbalance experiments.**

**The train/test split uses:**

```text
Training Set → 80%
Testing Set  → 20%
```

**with stratification to preserve the class distribution.**

---

# ⚠️ Underfitting vs Overfitting

**The project investigates the effect of Logistic Regression's regularization strength.**

### 🔵 Underfitting

**A very small value of:**

```text
C = 0.001
```

**is used to create a strongly regularized model.**

Strong regularization restricts model complexity and can lead to **underfitting**.

### 🔴 Overfitting

A large value of:

```text
C = 1000
```

**is used to create a weakly regularized model.**

Weak regularization allows the model greater flexibility and can increase the risk of **overfitting**.

---

# 📈 Learning Curve

A 5-fold cross-validation learning curve is generated using **F1-score**.

The curve compares:

* Training F1-score
* Validation F1-score
* Increasing training-set size

### Purpose

**The learning curve helps diagnose:**

* Underfitting
* Overfitting
* Generalization behavior
* Whether additional training data may improve performance

---

# 🛡️ Regularization Experiments

**Three regularization approaches are compared.**

## 1. L1 Regularization

**L1 regularization encourages sparse model coefficients and can effectively perform feature selection.**

```text
L1 → Sparse Features → Remove Less Useful Features
```

### Result

**F1 = 0.518**

**This was the best-performing regularization approach in the notebook.**

---

## 2. L2 Regularization

**L2 regularization shrinks model coefficients while generally retaining all features.**

### Result

**F1 = 0.435**

**The notebook interprets this as stable performance but less effective than L1 for the highly correlated/noisy EEG representation.**

---

## 3. Elastic Net

**Elastic Net combines L1 and L2 penalties.**

### Result

**F1 = 0.426**

**In this experiment, Elastic Net did not outperform the individual regularization methods.**

---

# 🏆 Regularization Results

| Method         |  F1 Score | Performance |
| -------------- | --------: | ----------- |
| 🥇 L1          | **0.518** | Best        |
| 🥈 L2          | **0.435** | Moderate    |
| 🥉 Elastic Net | **0.426** | Lowest      |

The notebook concludes that **L1 regularization provides the strongest performance**, suggesting that feature sparsity is beneficial for this EEG classification task.

---

# ⚖️ Class Imbalance Handling

**The project evaluates three different approaches to class imbalance.**

### 1. Class Weighting

```python
LogisticRegression(class_weight='balanced')
```

**This approach changes the importance assigned to different classes without changing the original dataset.**

---

### 2. SMOTE

**Synthetic Minority Over-sampling Technique (SMOTE)** creates synthetic minority-class observations.

```text
Original Training Data
        ↓
      SMOTE
        ↓
Balanced Training Data
        ↓
Logistic Regression
```

**The notebook reports that SMOTE slightly improves PR-AUC, although the improvement is limited.**

---

### 3. Random Undersampling

**Random Undersampling reduces the number of majority-class samples.**

```text
Majority Class ──┐
                 ├──→ Balanced Dataset
Minority Class ──┘
```

**The notebook reports that undersampling achieved the highest F1-score among the imbalance approaches, but at the cost of removing some majority-class information.**

---

# 📊 Evaluation Metrics

**The project focuses primarily on:**

## F1 Score

**F1-score combines:**

* Precision
* Recall

**It is especially useful when both false positives and false negatives matter.**

---

## PR-AUC

**Precision-Recall Area Under the Curve (PR-AUC)** evaluates the model's ability to rank positive examples effectively.

**It is particularly useful when evaluating binary classification problems where class distributions may not be perfectly balanced.**

---

# 🏆 Final Experimental Findings

**The experiments lead to the following main conclusions:**

### 🥇 Best Preprocessing

**Pipeline A**

F1 = **0.43**
PR-AUC = **0.57**

### 🥇 Best Regularization

**L1 Regularization**

F1 = **0.518**

### ⚖️ Class Imbalance

**The notebook reports that class-imbalance techniques produced only marginal improvements, suggesting that preprocessing and feature representation had a greater influence on model performance.**

---

# 💡 Key Insight

**The most important finding of the project is:**

> **Preserving useful EEG signal structure while applying controlled feature selection performs better than aggressive feature compression.**

Pipeline B reduced the original EEG representation to only a small number of statistical features before applying PCA. According to the notebook's analysis, this caused information loss and reduced classification performance.

The experiments further indicate that **L1 regularization is particularly useful for this type of high-dimensional EEG data**, because it can remove redundant or less informative features.

---

# 🧰 Technologies & Libraries

| Technology             | Purpose                  |
| ---------------------- | ------------------------ |
| 🐍 Python              | Programming language     |
| 🐼 Pandas              | Data manipulation        |
| 🔢 NumPy               | Numerical computation    |
| 🤖 Scikit-learn        | Machine learning         |
| 🧠 Logistic Regression | Classification           |
| 📏 StandardScaler      | Feature scaling          |
| 🔍 VarianceThreshold   | Feature selection        |
| 📉 PCA                 | Dimensionality reduction |
| ⚖️ SMOTE               | Oversampling             |
| 📊 Matplotlib          | Visualization            |
| 📓 Jupyter Notebook    | Development environment  |

---

# ⭐ Project Highlights

```text
🧠 EEG Signal Analysis
        ↓
🧹 Data Preprocessing
        ↓
📏 Feature Scaling
        ↓
🔇 Denoising
        ↓
🔍 Feature Selection
        ↓
📉 Dimensionality Reduction
        ↓
🤖 Logistic Regression
        ↓
🛡️ Regularization
        ↓
⚖️ Class Imbalance
        ↓
📊 F1 + PR-AUC
        ↓
🏆 Best Model Selection
```

---

# 📁 Project Structure

```text
EEG-Seizure-Classification/
│
├── shahid-assignment.ipynb
│
├── README.md
│
└── images/
    ├── pipeline_comparison.png
    ├── learning_curve.png
    ├── regularization_comparison.png
    ├── class_imbalance_comparison.png
    └── confusion_matrix.png
```

---

# 🚀 How to Run

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
```

### 3. Open the Notebook

```text
shahid-assignment.ipynb
```

Open it using:

* Jupyter Notebook
* JupyterLab
* Visual Studio Code
* Google Colab
* Kaggle Notebook

### 4. Run the Notebook

**Run the cells sequentially to reproduce:**

* Data loading
* Preprocessing
* Feature selection
* PCA
* Logistic Regression
* Learning curves
* Regularization experiments
* Class-imbalance experiments
* Performance evaluation

> **Note:** The notebook currently references Kaggle dataset paths such as `/kaggle/input/...`. If you run it outside Kaggle, these dataset paths need to be updated.

---

# 🎓 Learning Outcomes

**This project demonstrates practical experience with:**

* EEG data analysis
* Machine learning classification
* Data preprocessing
* Feature scaling
* Signal smoothing
* Feature selection
* PCA
* Logistic Regression
* Regularization
* Underfitting and overfitting
* Learning curves
* Class imbalance
* SMOTE
* Random undersampling
* Model evaluation
* F1-score
* PR-AUC

---
# 📌 Main Notebook

**The complete implementation, experiments, outputs, and analysis are available in:**

**`shahid-assignment(1).ipynb`**


# 👨‍💻 About The Developer
## Shahid Azam

**MS Computer Science Student**
**Specialization in Artificial Intelligence**
- 🎯 **Focus Areas:** Data Visualization, AI/ML, Big Data Mining

Institute of Management Sciences, Peshawar, Pakistan

I enjoy building complete software systems—from networking and operating system concepts to backend architecture, machine learning, and distributed systems.

- 🎓 **Degree / Field:** MS in Computer Science (Specialization in Artificial Intelligence)
- 🎓 **Degree / Field:** MS in Computer Science (Specialization in Artificial Intelligence)
- 🎯 **Focus Areas:** Data Visualization, AI/ML, Big Data Mining
- 📬 **Contact:** shahidazam2020@gmail.com
- <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" width="18" height="18" valign="middle" /> WhatsApp: <a href="https://wa.me/+923412772594" target="_blank">Chat on WhatsApp</a>

Skills demonstrated:

**Python • Machine Learning • EEG Analysis • Data Preprocessing • Feature Engineering • Scikit-learn • Logistic Regression • Regularization • PCA • SMOTE • Data Analysis**

---
### Areas of Interest

• Artificial Intelligence
• Machine Learning
• Deep Learning
• Natural Language Processing
• Data Analysis
• Data Visualization
• EEG Signal Analysis

# 🤝 Connect With Me

<p align="center">
<a href="https://github.com/shahidazam2020-oss">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
</a>
  <a href="https://www.linkedin.com/in/shahid-azam-mughal-787b58235">
<img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
</a>
  <a href="mailto:shahidazam2020@gmail.com">
<img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" />
</a>
  </p>

---
## ⭐ Project Highlights

```text
EEG Data Analysis
        +
Data Preprocessing
        +
Feature Engineering
        +
Machine Learning
        +
Regularization
        +
Class Imbalance Analysis
        =
EEG Classification Study
```

---
# ⭐ Repository
**If you find this project useful for learning or experimentation, consider giving the repository a ⭐.**

<p align="center">
**⭐ Star this repository • 🍴 Fork it • 🧠 Explore the code • 🚀 Build something new**
</p>

---
## 📜 Disclaimer

This project is intended for **academic and educational purposes**. The results demonstrate machine-learning experiments on the datasets used in the notebook and should not be interpreted as a clinically validated seizure-detection system.

---
# ⭐ Support

**If you find this repository useful for learning Machine Learning, consider giving it a ⭐.**

**Your feedback, suggestions, and contributions are welcome.**

