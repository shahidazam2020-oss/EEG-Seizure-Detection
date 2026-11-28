
<div align="center">
  <a href="https://github.com/shahidazam2020-oss/EEG-Seizure-Detection/stargazers">
    <img src="https://img.shields.io/github/stars/shahidazam2020-oss/EEG-Seizure-Detection?style=social&cacheSeconds=60" alt="GitHub Stars">
  </a>
  <a href="https://github.com/shahidazam2020-oss/EEG-Seizure-Detection/forks">
    <img src="https://img.shields.io/github/forks/shahidazam2020-oss/EEG-Seizure-Detection?style=social&cacheSeconds=60" alt="GitHub Forks">
  </a>
  <a href="https://github.com/shahidazam2020-oss/EEG-Seizure-Detection/commits">
    <img src="https://img.shields.io/github/last-commit/shahidazam2020-oss/EEG-Seizure-Detection?style=flat-square" alt="Last Commit">
  </a>
</div>

---
<div align="center">

# 🧠 EEG-BASED SEIZURE ANALYSIS & CLASSIFICATION

### Machine Learning | EEG Signal Processing | Feature Engineering

**An Experimental Machine Learning Framework for EEG Data Analysis and Classification**

**Exploring data preprocessing, feature selection, dimensionality reduction, regularization, and class-imbalance strategies for EEG-related classification tasks.**

<br>

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=for-the-badge&logo=numpy)](https://numpy.org/)


<br>

[📂 Repository](https://github.com/shahidazam2020-oss/EEG-Seizure-Detection) •
[📓 Main Notebook](https://github.com/shahidazam2020-oss/EEG-Seizure-Detection/blob/main/shahid-assignment.ipynb) •
[👨‍💻 Developer](https://github.com/shahidazam2020-oss)

</div>

---

## 📌 Table of Contents

- [📖 About the Project](#-about-the-project)
- [🎯 Project Objectives](#-project-objectives)
- [✨ Key Features](#-key-features)
- [📊 Datasets](#-datasets)
- [🔬 Machine Learning Workflow](#-machine-learning-workflow)
- [🧹 Data Preprocessing](#-data-preprocessing)
- [🔍 Feature Engineering](#-feature-engineering)
- [📉 Dimensionality Reduction](#-dimensionality-reduction)
- [🤖 Machine Learning Model](#-machine-learning-model)
- [🛡️ Regularization Experiments](#️-regularization-experiments)
- [⚖️ Class Imbalance Handling](#️-class-imbalance-handling)
- [📈 Model Evaluation](#-model-evaluation)
- [🏆 Experimental Findings](#-experimental-findings)
- [🛠️ Technology Stack](#️-technology-stack)
- [📂 Repository Structure](#-repository-structure)
- [⚙️ Installation and Setup](#️-installation-and-setup)
- [▶️ Running the Project](#️-running-the-project)
- [⚠️ Limitations](#️-limitations)
- [🚀 Future Development Roadmap](#-future-development-roadmap)
- [🤝 Contributing](#-contributing)
- [📚 Learning Outcomes](#-learning-outcomes)
- [👨‍💻 About the Developer](#-about-the-developer)
- [📬 Connect With Me](#-connect-with-me)
- [⭐ Support the Project](#-support-the-project)

---

## 📖 About the Project

**EEG-Based Seizure Analysis and Classification** is a Python-based machine learning project focused on investigating EEG-related datasets through preprocessing, feature engineering, statistical analysis, and supervised classification.

**The project explores how different data preparation and modeling strategies influence classification performance.**

The primary classification algorithm is **Logistic Regression**, supported by experiments involving:

- Feature standardization.
- EEG feature smoothing and denoising.
- Variance-based feature selection.
- Statistical feature extraction.
- Principal Component Analysis (PCA).
- L1 and L2 regularization.
- Elastic Net regularization.
- Class weighting.
- Synthetic Minority Over-sampling Technique (SMOTE).
- Random undersampling.

**The project also investigates the effects of underfitting, overfitting, model complexity, and class imbalance using classification metrics such as F1-score and Precision-Recall AUC.**

### 💡 Project Vision

**To investigate how preprocessing, feature representation, and machine learning optimization techniques affect classification performance in EEG-related datasets.**

---

## 🎯 Project Objectives

**The main objectives include:**

- Analyze EEG-related datasets.
- Understand the statistical characteristics of EEG features.
- Perform data cleaning and preprocessing.
- Handle missing values.
- Standardize numerical features.
- Investigate signal smoothing and denoising.
- Identify informative features.
- Explore dimensionality reduction using PCA.
- Implement Logistic Regression.
- Investigate model underfitting and overfitting.
- Compare L1, L2, and Elastic Net regularization.
- Evaluate different class-imbalance strategies.
- Compare preprocessing pipelines.
- Evaluate models using F1-score and PR-AUC.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 EEG Data Analysis | Exploration of EEG-related datasets and features. |
| 🧹 Data Preprocessing | Missing-value handling and dataset preparation. |
| 📏 Standardization | Feature scaling using StandardScaler. |
| 🔇 Signal Smoothing | Rolling-window mean-based smoothing. |
| 🔍 Feature Selection | Low-variance feature removal. |
| 📊 Statistical Features | Mean, standard deviation, maximum, minimum, and range extraction. |
| 📉 Dimensionality Reduction | PCA with 95% variance retention. |
| 🤖 Classification | Logistic Regression implementation. |
| 🛡️ Regularization | L1, L2, and Elastic Net comparisons. |
| ⚖️ Class Imbalance | Class weighting, SMOTE, and random undersampling. |
| 🔄 Cross-Validation | Five-fold cross-validation for learning-curve analysis. |
| 📈 Model Evaluation | F1-score and Precision-Recall AUC. |
| 📊 Pipeline Comparison | Evaluation of different preprocessing strategies. |

---

## 📊 Datasets

**The repository includes three EEG-related datasets.**

### 1. EEG Feature Dataset

**Used to explore EEG features and their statistical properties.**

### 2. Epileptic Seizure Recognition Dataset

**Used for seizure-related classification experiments.**

**The target variable is transformed into a binary classification format:**

| Original Class | Binary Label |
|---|---|
| Seizure | 1 |
| Non-Seizure | 0 |

### 3. EEG Eye State Dataset

**This dataset is used in the notebook's main classification pipeline with the `eyeDetection` variable as its target.**

**The feature matrix is created by separating the target column:**

```python
X = df3.drop(columns=["eyeDetection"])
y = df3["eyeDetection"]
```

**Important:** EEG Eye State classification and epileptic seizure classification are different prediction tasks. Their datasets and results should be documented separately when presenting research findings.

---

## 🔬 Machine Learning Workflow

**The project follows a structured experimental workflow.**

```text
                  🧠 EEG DATASETS
                         │
                         ▼
                📥 DATA LOADING
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
         Denoising           Statistical Features
              │                     │
       Variance Selection           │
              │                     ▼
              │                    PCA
              │                     │
              └──────────┬──────────┘
                         ▼
                🤖 LOGISTIC REGRESSION
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
             L1         L2       Elastic Net
              │          │          │
              └──────────┼──────────┘
                         ▼
                 ⚖️ CLASS BALANCING
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          SMOTE    Undersampling  Class Weights
             │           │           │
             └───────────┼───────────┘
                         ▼
                  📈 EVALUATION
                         │
                   F1 / PR-AUC
                         │
                         ▼
                🏆 MODEL COMPARISON
```

---

## 🧹 Data Preprocessing

**Data preprocessing prepares the EEG feature representation for machine learning.**

### Missing Value Handling

**Missing numerical values are replaced using the corresponding feature mean.**

```python
X = X.fillna(X.mean())
```

### Feature Standardization

**The project uses `StandardScaler` to standardize numerical features.**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

**Standardization is particularly important for algorithms and transformations sensitive to feature scales, including Logistic Regression and PCA.**

### EEG Feature Smoothing

**A rolling-window mean operation is used to reduce short-term fluctuations in the feature representation.**

```python
X_denoised = X_scaled_df.rolling(
    window=3,
    min_periods=1
).mean()
```

**This operation is applied to the feature representation; it should not be confused with a complete raw EEG signal-processing pipeline.**

---

## 🔍 Feature Engineering

**The project investigates two different feature-processing approaches.**

### 🔵 Pipeline A — Feature Selection

**Pipeline A preserves the original feature representation while removing low-variance features.**

**Technique:** `VarianceThreshold`

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)

X_selected = selector.fit_transform(X_denoised)
```

#### Workflow

```text
Raw EEG Features
       │
       ▼
Standardization
       │
       ▼
Smoothing / Denoising
       │
       ▼
Variance Threshold
       │
       ▼
Selected Features
       │
       ▼
Logistic Regression
```

**Objective:** Retain useful feature information while eliminating features with very low variance.

### 🟣 Pipeline B — Statistical Feature Extraction

**Pipeline B extracts statistical characteristics from each observation.**

**The extracted features include:**

- Mean
- Standard deviation
- Maximum
- Minimum
- Range

Example:

```python
X_features = pd.DataFrame({
    "mean": X_df.mean(axis=1),
    "std": X_df.std(axis=1),
    "max": X_df.max(axis=1),
    "min": X_df.min(axis=1),
    "range": X_df.max(axis=1) - X_df.min(axis=1)
})
```

**The resulting features are standardized and processed using PCA.**

---

## 📉 Dimensionality Reduction

**Principal Component Analysis (PCA) is investigated as a dimensionality reduction technique.**

### PCA Configuration

**The notebook uses:**

**Variance Retention: 95%**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_features_scaled)
```

### Purpose

**PCA transforms the original feature space into a smaller set of principal components while retaining a specified proportion of variance.**

### Pipeline B

```text
Raw EEG Features
       │
       ▼
Statistical Feature Extraction
       │
       ▼
Mean / Std / Max / Min / Range
       │
       ▼
Standardization
       │
       ▼
PCA (95% Variance)
       │
       ▼
Logistic Regression
```

---

## 🤖 Machine Learning Model

### Logistic Regression

**Logistic Regression is the primary supervised classification algorithm used in the project.**

**It estimates the probability of a binary outcome using a logistic function.**

\[
P(y=1|x)=\frac{1}{1+e^{-z}}
\]

Where:

\[
z=w^Tx+b
\]

### Model Training

**The notebook uses an 80/20 training and testing split with stratification.**

| Dataset Portion | Percentage |
|---|---:|
| Training | 80% |
| Testing | 20% |

**Stratification helps preserve the target-class distribution across the training and testing sets.**

### Model Experiments

**The project investigates:**

- Baseline Logistic Regression.
- Strong regularization.
- Weak regularization.
- L1 regularization.
- L2 regularization.
- Elastic Net regularization.
- Class-weighted classification.
- SMOTE-based training.
- Random undersampling.

---

## 🛡️ Regularization Experiments

**Regularization controls model complexity and helps investigate the balance between underfitting and overfitting.**

### 1. L1 Regularization

**L1 regularization encourages sparse model coefficients.**

**It can reduce the influence of less informative features.**

**Reported F1-score: 0.518**

### 2. L2 Regularization

**L2 regularization penalizes large coefficients and generally retains all model features.**

**Reported F1-score: 0.435**

### 3. Elastic Net

**Elastic Net combines L1 and L2 regularization.**

**Reported F1-score: 0.426**

### Regularization Comparison

| Method | Reported F1-score |
|---|---:|
| 🥇 L1 | 0.518 |
| 🥈 L2 | 0.435 |
| 🥉 Elastic Net | 0.426 |

**According to the notebook's reported experiments, L1 regularization achieved the highest F1-score among these three methods.**

---

## ⚖️ Class Imbalance Handling

**Class imbalance can influence classification performance, particularly when one target class has substantially more observations than another.**

**The project investigates three approaches.**

### 1. Class Weighting

```python
LogisticRegression(class_weight="balanced")
```

**This approach adjusts the importance assigned to different classes during model training.**

### 2. SMOTE

**Synthetic Minority Over-sampling Technique generates synthetic minority-class observations.**

```text
Original Training Data
         │
         ▼
        SMOTE
         │
         ▼
Balanced Training Data
         │
         ▼
Logistic Regression
```

**The notebook reports a limited improvement in PR-AUC from SMOTE.**

### 3. Random Undersampling

**Random undersampling reduces the number of majority-class observations.**

```text
Majority Class ──┐
                 ├──► Balanced Training Dataset
Minority Class ──┘
```

**The notebook reports that undersampling achieved the highest F1-score among the evaluated imbalance-handling approaches, although it reduces the amount of majority-class information available for training.**

---

## 📈 Model Evaluation

**The project focuses on two primary evaluation metrics.**

### F1-Score

**F1-score is the harmonic mean of precision and recall.**

\[
F1=2\times\frac{Precision\times Recall}{Precision+Recall}
\]

**It is useful when both false positives and false negatives matter.**

### Precision-Recall AUC

**PR-AUC evaluates the relationship between precision and recall across classification thresholds.**

**It is particularly useful for investigating classification performance under class imbalance.**

### Learning Curve Analysis

**The notebook also generates a five-fold cross-validation learning curve.**

It compares:

- Training F1-score.
- Validation F1-score.
- Increasing training-set size.

**This analysis helps investigate model generalization, underfitting, and overfitting.**

---

## 🏆 Experimental Findings

**The existing notebook reports the following results.**

### Preprocessing Pipeline Comparison

| Pipeline | Approach | F1-score | PR-AUC |
|---|---|---:|---:|
| 🔵 Pipeline A | Denoising + Variance Selection | 0.43 | 0.57 |
| 🟣 Pipeline B | Statistical Features + PCA | 0.21 | 0.48 |

**Pipeline A performed better than Pipeline B in the reported experiment.**

**The notebook suggests that reducing the feature representation to a small number of statistical features may have removed useful information.**

### Regularization Comparison

| Regularization | F1-score |
|---|---:|
| L1 | 0.518 |
| L2 | 0.435 |
| Elastic Net | 0.426 |

**L1 regularization achieved the strongest reported result among the three regularization approaches.**

### Main Experimental Insight

> Preserving informative feature representations while applying controlled feature selection may be more effective than aggressive feature compression.

**Research interpretation:** The reported experiments suggest that feature representation and preprocessing choices can have a substantial influence on classification performance.

**These values are experimental results reported in the existing notebook, not evidence of clinical diagnostic accuracy.**

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| Jupyter Notebook | Interactive experimentation |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| Logistic Regression | Classification |
| StandardScaler | Feature standardization |
| VarianceThreshold | Feature selection |
| PCA | Dimensionality reduction |
| Imbalanced-learn | Class imbalance handling |
| SMOTE | Synthetic oversampling |
| Matplotlib | Data visualization |

### Technology Badges

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)

</div>

---

## 📂 Repository Structure

**The repository currently contains the following important files:**

```text
EEG-Seizure-Detection/
│
├── README.md
│
├── EEG-Eye-State.csv
│
├── Epileptic Seizure Recognition.csv
│
├── features_raw.csv
│
├── shahid-assignment.ipynb
│
├── Q2_Emotion_Detection.ipynb
│
├── Major Assignment.rar
│
├── Shahid Presentation File.pptx
│
├── Shahid Python Work.docx
│
└── Shahid Work.pdf
```

**This structure reflects the existing repository at the time of documentation. Additional files may be present in the archive or added later.**

---

## ⚙️ Installation and Setup

### Prerequisites

Make sure you have installed:

- Python 3.x
- Jupyter Notebook
- pip package manager

### Step 1 — Clone the Repository

```bash
git clone https://github.com/shahidazam2020-oss/EEG-Seizure-Detection.git
```

### Step 2 — Navigate to the Project Directory

```bash
cd EEG-Seizure-Detection
```

### Step 3 — Install Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn jupyter
```

### Step 4 — Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## ▶️ Running the Project

Open the main notebook:

```text
shahid-assignment.ipynb
```

**Run the cells sequentially to explore:**

1. Dataset loading.
2. Data inspection.
3. Missing-value handling.
4. Feature standardization.
5. EEG feature smoothing.
6. Feature selection.
7. Statistical feature extraction.
8. PCA.
9. Logistic Regression.
10. Learning curve analysis.
11. Regularization experiments.
12. Class-imbalance handling.
13. Model evaluation.
14. Experimental comparison.

### Dataset Path Configuration

**The existing notebook contains Kaggle-specific dataset paths.**

**If you are running the project locally, update the dataset paths to match your local directory.**

Example:

```python
import pandas as pd

df = pd.read_csv("Epileptic Seizure Recognition.csv")
```

---

## ⚠️ Limitations

The project is an experimental machine learning study and has several limitations that should be considered.

- The repository contains multiple EEG-related datasets with different prediction targets.
- EEG Eye State classification should not be interpreted as epileptic seizure detection.
- The experiments use feature-based classification rather than a complete raw multichannel EEG seizure-monitoring system.
- The reported results are dependent on the datasets and experimental setup.
- Further patient-independent validation is needed for seizure-related research.
- More extensive evaluation is needed before drawing conclusions about clinical applicability.
- Data preprocessing and feature transformations should be fitted on training data only to avoid information leakage.

---

## 🚀 Future Development Roadmap

### Phase 1 — Research Documentation

- [ ] Clearly separate seizure classification and eye-state classification experiments.
- [ ] Add dataset source references.
- [ ] Document the preprocessing methodology.
- [ ] Explain the mathematical foundations of each algorithm.
- [ ] Add a reproducibility guide.

### Phase 2 — Advanced Machine Learning

- [ ] Compare Logistic Regression with Random Forest.
- [ ] Explore Support Vector Machines.
- [ ] Investigate Gradient Boosting.
- [ ] Experiment with XGBoost.
- [ ] Introduce systematic hyperparameter optimization.

### Phase 3 — Evaluation Improvements

- [ ] Add confusion matrices.
- [ ] Include sensitivity and specificity.
- [ ] Add ROC-AUC analysis.
- [ ] Include classification reports.
- [ ] Perform patient-independent validation for seizure experiments.
- [ ] Investigate data leakage and class-distribution effects.

### Phase 4 — Deep Learning and Signal Processing

- [ ] Explore CNN-based EEG classification.
- [ ] Investigate LSTM-based temporal modeling.
- [ ] Explore CNN-LSTM architectures.
- [ ] Study EEG frequency-band features.
- [ ] Investigate time-frequency representations.
- [ ] Explore explainable AI for EEG classification.

### Phase 5 — Professional Project Development

- [ ] Organize notebooks and datasets into dedicated folders.
- [ ] Add `requirements.txt`.
- [ ] Add experiment configuration files.
- [ ] Store evaluation results separately.
- [ ] Add automated testing.
- [ ] Improve visualization and documentation.

---

## 🤝 Contributing

Contributions, suggestions, and research improvements are welcome.

### How to Contribute

1. Fork the repository.
2. Create a new branch.
3. Implement your changes.
4. Test your code.
5. Update relevant documentation.
6. Submit a pull request.

### Contribution Guidelines

- Write clean and readable Python code.
- Use meaningful variable names.
- Document preprocessing and modeling decisions.
- Maintain reproducibility.
- Avoid committing unnecessary files.
- Include references for external datasets and research methods.

---

## 📚 Learning Outcomes

This project provides practical exposure to:

- EEG-related data analysis.
- Python programming.
- Data cleaning and preprocessing.
- Feature standardization.
- Signal smoothing.
- Feature engineering.
- Feature selection.
- Principal Component Analysis.
- Logistic Regression.
- Model regularization.
- Underfitting and overfitting.
- Learning curve analysis.
- Class-imbalance handling.
- SMOTE.
- Random undersampling.
- F1-score and PR-AUC evaluation.
- Experimental machine learning comparison.

---

## ⚠️ Medical and Research Disclaimer

This repository is intended for educational, academic, and machine learning research purposes.

The experiments are not a clinically validated seizure-detection system and must not be used for medical diagnosis, treatment decisions, or patient management.

Clinical applications require appropriate datasets, rigorous validation, expert review, and regulatory approval.

---

## 👨‍💻 About the Developer

<div align="center">

### Shahid Azam

**MS Computer Science | Artificial Intelligence Specialization**

Institute of Management Sciences, Peshawar, Pakistan

</div>

My interests include Artificial Intelligence, Machine Learning, Deep Learning, Natural Language Processing, Computer Vision, and research-oriented software development.

### Areas of Interest

- 🧠 Artificial Intelligence
- 🤖 Machine Learning
- 🔬 Deep Learning
- 🏥 Healthcare AI
- 📊 Data Analysis
- 🧬 Biomedical Signal Analysis
- 📚 Academic Research

---

## 📬 Connect With Me

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/shahidazam2020-oss)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/shahid-azam-mughal-787b58235)

[![ResearchGate](https://img.shields.io/badge/ResearchGate-Follow-green?style=for-the-badge&logo=researchgate)](https://www.researchgate.net/profile/Shahid-Azam-5)

[![ORCID](https://img.shields.io/badge/ORCID-Connect-green?style=for-the-badge&logo=orcid)](https://orcid.org/0009-0001-6825-8863)

</div>

---

## ⭐ Support the Project

If you find this repository useful for learning, research, or machine learning experimentation:

- ⭐ Star the repository.
- 🍴 Fork the project.
- 🐛 Report issues.
- 💡 Suggest improvements.
- 🤝 Contribute new ideas.
- 📢 Share the project with other researchers and developers.

Your support encourages collaboration, continuous learning, and further development.

---

<div align="center">

### 🧠 Exploring EEG Signals Through Machine Learning

**Analyze • Preprocess • Engineer • Classify • Evaluate**

Developed with ❤️ by **Shahid Azam**

[⬆ Back to Top](#-eeg-based-seizure-analysis--classification)

</div>
