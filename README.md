Breast Cancer Classification
A machine learning project that compares five classification algorithms on the Breast Cancer Wisconsin (Diagnostic) dataset to predict whether a tumour is benign or malignant.
---
Table of Contents
1. Problem Statement
2. Dataset Description
3. Models and Evaluation Metrics
4. Model Performance
5. Performance Analysis
6. Overall Conclusion
7. Project Structure
8. Installation and Usage
9. Streamlit Application
10. Repository
---
1. Problem Statement
The objective of this project is to build and evaluate supervised machine learning classifiers that predict whether a tumour is benign (B) or malignant (M) using measurements of cell nuclei.
The following five classification algorithms are implemented and evaluated on the same training and testing split:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Support Vector Machine (SVM)
2. Dataset Description
Property	Description
Dataset	Breast Cancer Wisconsin (Diagnostic)
Source	UCI Machine Learning Repository
Total instances	569
Original columns	32
Input features used	30 numerical features
Target column	`diagnosis`
Target classes	Benign (`B`) and Malignant (`M`)
Missing values	0
Data preparation
The `id` column is removed because it is an identifier, not a predictive feature.
The target variable is encoded as:
Benign (`B`) = `0`
Malignant (`M`) = `1`
The dataset is split into 80% training data and 20% testing data.
The split contains 455 training samples and 114 testing samples.
Stratification is used to preserve the class proportions in both subsets.
3. Models and Evaluation Metrics
The five models are evaluated on the held-out test set using the following metrics:
Metric	What it measures
Accuracy	Proportion of all predictions that are correct.
AUC	How well the model distinguishes between the two classes across classification thresholds.
Precision	Of the cases predicted as malignant, the proportion that are actually malignant.
Recall	Of the actual malignant cases, the proportion correctly identified.
F1 Score	Harmonic mean of precision and recall.
MCC	Matthews Correlation Coefficient; a balanced measure that accounts for true and false predictions across both classes.
> **Note:** Since malignant cases are encoded as `1`, precision and recall refer to the model's performance when identifying malignant cases.
4. Model Performance
The following results were obtained using the 114-sample test set.
Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.9649	0.9960	0.9750	0.9286	0.9512	0.9245
Decision Tree	0.9298	0.9246	0.9048	0.9048	0.9048	0.8492
KNN	0.9561	0.9823	0.9744	0.9048	0.9383	0.9058
Gaussian Naive Bayes	0.9386	0.9934	1.0000	0.8333	0.9091	0.8715
Support Vector Machine	0.9737	0.9947	1.0000	0.9286	0.9630	0.9442
5. Performance Analysis
Logistic Regression
Logistic Regression achieved strong overall classification performance and the highest AUC among the five models. This indicates that it distinguishes benign and malignant cases very effectively across different classification thresholds. Its high precision and recall also show that it identifies most malignant cases while producing few false-positive predictions.
Decision Tree
Decision Tree had the lowest accuracy, AUC, and MCC among the five models in this experiment. Its precision and recall were equal, indicating similar performance in avoiding false positives and identifying malignant cases. Its comparatively lower results may reflect the sensitivity of a single decision tree to the training data and its chosen splitting rules.
K-Nearest Neighbors (KNN)
KNN achieved strong accuracy and AUC. Its high precision indicates that most cases predicted as malignant were actually malignant. Its recall was lower than its precision, meaning that some malignant cases were classified as benign. This suggests that distance-based similarity is useful for this dataset, although some borderline cases may be difficult to classify.
Gaussian Naive Bayes
Gaussian Naive Bayes achieved perfect precision on the test set: every case it predicted as malignant was actually malignant. However, it had the lowest recall, meaning that it missed a greater proportion of actual malignant cases. Its performance may be affected by the Naive Bayes assumption that input features are conditionally independent, whereas several measurements in this dataset may be correlated.
Support Vector Machine (SVM)
SVM achieved the highest accuracy, F1 score, and MCC. Its perfect precision means that it produced no false-positive malignant predictions on this test set, while its recall indicates that it identified most actual malignant cases. Its high AUC also shows strong class discrimination. Feature scaling was included in the SVM pipeline, which is important because SVM is sensitive to differences in feature magnitudes.
6. Overall Conclusion
All five classifiers demonstrated good performance on the Breast Cancer Wisconsin dataset, although their ability to distinguish between benign and malignant cases varied. SVM provided the strongest overall balance across the evaluation metrics, while Logistic Regression also performed exceptionally well in distinguishing between the two classes. KNN showed consistent performance, whereas Naive Bayes was more conservative in predicting malignant cases and Decision Tree had comparatively weaker results. Based on this test-set comparison, SVM is the best-performing model for this experiment, with Logistic Regression serving as a strong alternative.
These findings are based on one stratified train-test split and should not be interpreted as a guarantee that the same model will perform best on every unseen dataset.
7. Project Structure
```text
BITS-ML-TASK/
├── train_models.py
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```
8. Installation and Usage
Install the dependencies
```bash
pip install -r requirements.txt
```
Train the models
Place the original `wdbc.data` dataset file at the path configured in `train_models.py`, then run:
```bash
python train_models.py
```
The training script trains the five classifiers, evaluates them, saves the trained models in the `model/` directory, and exports the held-out test data to `test_data.csv`.
Run the Streamlit application
Once `app.py` is available, launch the application with:
```bash
streamlit run app.py
```
9. Streamlit Application
Live application: Add the deployed Streamlit app URL here after deployment.
The application is designed to provide:
CSV upload for test data.
A dropdown to select one of the five trained classifiers.
Display of the evaluation metrics.
A confusion matrix or classification report.
The application should load the saved `.joblib` model selected by the user and evaluate the uploaded test data without retraining the model.
10. Repository
GitHub repository: BITS-ML-TASK
---
