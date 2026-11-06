
# Breast Cancer Classification

Machine learning project comparing five classifiers on the **Breast Cancer Wisconsin (Diagnostic)** dataset to predict whether a tumour is **benign** or **malignant**.

---

## Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Description](#2-dataset-description)
3. [Models and Evaluation Metrics](#3-models-and-evaluation-metrics)
4. [Model Comparison](#4-model-comparison)
5. [Performance Observations](#5-performance-observations)
6. [Overall Conclusion](#6-overall-conclusion)
7. [Repository Structure](#7-repository-structure)
8. [Installation and Execution](#8-installation-and-execution)
9. [Streamlit Application](#9-streamlit-application)

---

## 1. Problem Statement

The objective is to build supervised machine learning models that classify a tumour as **benign (B)** or **malignant (M)** using measurements of cell nuclei.

The five implemented classifiers are:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

All five models are trained and evaluated using the same training and testing split.

## 2. Dataset Description

<table>
<thead>
<tr><th>Property</th><th>Description</th></tr>
</thead>
<tbody>
<tr><td>Dataset</td><td>Breast Cancer Wisconsin (Diagnostic)</td></tr>
<tr><td>Source</td><td><a href="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic">UCI Machine Learning Repository</a></td></tr>
<tr><td>Instances</td><td>569</td></tr>
<tr><td>Original columns</td><td>32</td></tr>
<tr><td>Input features used</td><td>30</td></tr>
<tr><td>Target column</td><td>diagnosis</td></tr>
<tr><td>Classes</td><td>Benign (B) and Malignant (M)</td></tr>
<tr><td>Missing values</td><td>0</td></tr>
</tbody>
</table>

### Data Preprocessing

- The `id` column is removed because it is an identifier, not a predictive feature.
- Diagnosis labels are encoded as:
  - Benign (B) = 0
  - Malignant (M) = 1
- The data is split into 80% training and 20% testing data.
- The split contains 455 training samples and 114 testing samples.
- Stratification is used to preserve the class proportions in both sets.

## 3. Models and Evaluation Metrics

The models are evaluated on the held-out test set using these metrics:

<table>
<thead>
<tr><th>Metric</th><th>Meaning</th></tr>
</thead>
<tbody>
<tr><td><b>Accuracy</b></td><td>Proportion of all predictions that are correct.</td></tr>
<tr><td><b>AUC</b></td><td>How well the model distinguishes between the two classes across classification thresholds.</td></tr>
<tr><td><b>Precision</b></td><td>Of the cases predicted as malignant, the proportion that are actually malignant.</td></tr>
<tr><td><b>Recall</b></td><td>Of the actual malignant cases, the proportion correctly identified.</td></tr>
<tr><td><b>F1 Score</b></td><td>Harmonic mean of precision and recall.</td></tr>
<tr><td><b>MCC</b></td><td>Balanced measure of classification quality, accounting for true and false predictions in both classes.</td></tr>
</tbody>
</table>

*Since malignant is encoded as 1, precision and recall describe performance when identifying malignant cases.*

## 4. Model Comparison

Results on the 114-sample test set:

<table>
<thead>
<tr>
<th>ML Model</th>
<th>Accuracy</th>
<th>AUC</th>
<th>Precision</th>
<th>Recall</th>
<th>F1 Score</th>
<th>MCC</th>
</tr>
</thead>
<tbody>
<tr><td>Logistic Regression</td><td>0.9649</td><td><b>0.9960</b></td><td>0.9750</td><td>0.9286</td><td>0.9512</td><td>0.9245</td></tr>
<tr><td>Decision Tree</td><td>0.9298</td><td>0.9246</td><td>0.9048</td><td>0.9048</td><td>0.9048</td><td>0.8492</td></tr>
<tr><td>KNN</td><td>0.9561</td><td>0.9823</td><td>0.9744</td><td>0.9048</td><td>0.9383</td><td>0.9058</td></tr>
<tr><td>Gaussian Naive Bayes</td><td>0.9386</td><td>0.9934</td><td><b>1.0000</b></td><td>0.8333</td><td>0.9091</td><td>0.8715</td></tr>
<tr><td><b>Support Vector Machine</b></td><td><b>0.9737</b></td><td>0.9947</td><td><b>1.0000</b></td><td>0.9286</td><td><b>0.9630</b></td><td><b>0.9442</b></td></tr>
</tbody>
</table>

## 5. Performance Observations

### Logistic Regression

Logistic Regression achieved the highest AUC, indicating excellent class discrimination across different classification thresholds.

Its high precision and recall show that it identifies most malignant cases while producing few false-positive predictions. The strong F1 and MCC scores also indicate a good balance in its classification performance.

### Decision Tree

Decision Tree had the lowest accuracy, AUC, and MCC among the five models in this experiment.

Its equal precision and recall indicate a similar balance between false-positive predictions and correctly identified malignant cases. Its comparatively weaker results may be due to the sensitivity of a single tree to the training data and its splitting decisions.

### K-Nearest Neighbors (KNN)

KNN showed strong overall performance. Its high precision means that most cases predicted as malignant were actually malignant.

Its lower recall indicates that some malignant cases were classified as benign. This suggests that distance-based similarity is useful for this dataset, although borderline cases may be harder to classify.

### Gaussian Naive Bayes

Naive Bayes achieved 100% precision, meaning that every case it predicted as malignant was actually malignant in this test set.

However, it had the lowest recall, so it missed a greater proportion of actual malignant cases. Its performance may be affected by the Naive Bayes assumption that input features are conditionally independent, while some measurements in this dataset may be correlated.

### Support Vector Machine (SVM)

SVM achieved the strongest overall results, with the highest accuracy, F1 score, and MCC.

Its 100% precision indicates that it produced no false-positive malignant predictions in this test set. Its high recall shows that it also identified most actual malignant cases.

Feature scaling was included in the SVM pipeline because SVM is sensitive to differences in feature magnitudes.

## 6. Overall Conclusion

All five classifiers performed well, but their classification strengths differed.

**SVM was the best overall model** in this experiment because it provided the strongest balance across the evaluation metrics. Logistic Regression was also highly effective, particularly in distinguishing between the two classes.

KNN showed consistent performance. Naive Bayes was more conservative when predicting malignant cases, while Decision Tree had comparatively weaker results.

These findings are based on one stratified train-test split and do not guarantee that the same model will perform best on every unseen dataset.

## 7. Repository Structure

The repository is organised as follows:

```text
BITS-ML-TASK/
│
├── train_models.py
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

### File Descriptions

| File / Folder | Purpose |
|---|---|
| `train_models.py` | Loads the dataset, preprocesses it, trains and evaluates the five classifiers, saves the models, and exports test data. |
| `app.py` | Streamlit application for uploading test data, selecting a model, and viewing evaluation results. |
| `requirements.txt` | Lists the Python dependencies required to run the project. |
| `README.md` | Project documentation. |
| `test_data.csv` | Held-out test features and the true diagnosis labels. |
| `model/` | Contains the five separately saved trained model files. |

## 8. Installation and Execution

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train and Save the Models

Place `wdbc.data` at the dataset path configured in `train_models.py`, then run:

```bash
python train_models.py
```

The script trains all five classifiers, evaluates them, saves each model in the `model/` folder, and exports `test_data.csv`.

### Run the Streamlit Application

After `app.py` has been created, run:

```bash
streamlit run app.py
```

## 9. Streamlit Application

**Live Application:** (https://bits-ml-task-1.streamlit.app/)

The Streamlit application is designed to provide:

- CSV upload for test data.
- A dropdown to select one of the five trained classifiers.
- Display of the evaluation metrics.
- A confusion matrix or classification report.

The application loads the selected saved model and evaluates the uploaded test data **without retraining** the model.

---

## GitHub Repository

[**BITS-ML-TASK**](https://github.com/ManojKumarSingh31/BITS-ML-TASK)
