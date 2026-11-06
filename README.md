Breast Cancer Classification
a. Problem Statement
This project applies five supervised machine learning classification algorithms to the Breast Cancer Wisconsin (Diagnostic) dataset. The objective is to predict whether a tumour is benign (B) or malignant (M) using measurements of cell nuclei.

The five classifiers implemented are:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)
All five models are trained using the same training dataset and evaluated on the same held-out test dataset.
b. Dataset Description
- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Source: UCI Machine Learning Repository
- Total instances: 569
- Original columns: 32 (ID, diagnosis, and 30 numerical features)
- Predictive features used: 30
- Target column: diagnosis
- Classes: Benign (B) and Malignant (M)
- Missing values: 0
The ID column is excluded from model training because it is an identifier rather than a predictive measurement. The diagnosis labels are encoded as:
- Benign (B) = 0
- Malignant (M) = 1
The dataset is divided into 80% training data and 20% testing data, resulting in 455 training samples and 114 testing samples.
c. GitHub Repository Link
BITS-ML-TASK GitHub Repository
d. Models Used and Evaluation
The models are evaluated on the held-out test dataset using the following metrics:
- Accuracy: Proportion of predictions that are correct.
- AUC: Area under the ROC curve; measures how well the model distinguishes between the two classes.
- Precision: Proportion of predicted malignant cases that are actually malignant.
- Recall: Proportion of actual malignant cases correctly identified.
- F1 Score: Harmonic mean of precision and recall.
- MCC: Matthews Correlation Coefficient; a balanced measure based on true and false positives and negatives.
Model Comparison
ML Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.9649	0.9960	0.9750	0.9286	0.9512	0.9245
Decision Tree	0.9298	0.9246	0.9048	0.9048	0.9048	0.8492
KNN	0.9561	0.9823	0.9744	0.9048	0.9383	0.9058
Gaussian Naive Bayes	0.9386	0.9934	1.0000	0.8333	0.9091	0.8715
Support Vector Machine	0.9737	0.9947	1.0000	0.9286	0.9630	0.9442


Observations on Model Performance
ML Model	Observation about Performance
Logistic Regression	Achieved an accuracy of 96.49% and the highest AUC of 0.9960, indicating excellent ability to distinguish between benign and malignant tumours. Its precision (97.50%) and recall (92.86%) indicate that it correctly identifies most malignant cases while producing very few false positives. The high F1 score (0.9512) and MCC (0.9245) show a strong balance between classification performance and reliability. This suggests that the relationship between the input features and the diagnosis can be modelled effectively using a logistic decision boundary.
Decision Tree	Achieved an accuracy of 92.98%, the lowest among the five models. Its precision and recall are both 90.48%, indicating relatively balanced performance in identifying malignant cases and avoiding false positives. However, its lowest AUC (0.9246) and MCC (0.8492) suggest weaker overall discrimination and classification reliability. A single decision tree may be sensitive to the training data and its individual splitting decisions, resulting in less effective generalisation.
KNN	Achieved an accuracy of 95.61% and an AUC of 0.9823, indicating strong classification performance. Its precision of 97.44% shows that most cases predicted as malignant were actually malignant. However, its lower recall of 90.48% indicates that some malignant cases were misclassified as benign. Its F1 score (0.9383) and MCC (0.9058) demonstrate good overall performance. This suggests that similarity between neighbouring data points is useful for classification, although borderline cases may be harder to distinguish.
Gaussian Naive Bayes	Achieved a precision of 100%, meaning that every case predicted as malignant was actually malignant. However, its recall of 83.33% is the lowest among all five models, indicating that it misses more actual malignant cases. Its AUC of 0.9934 is high, suggesting that it can distinguish between the classes well based on predicted probabilities, even though its final classifications are less effective at the default threshold. The conditional independence assumption of Naive Bayes may also limit its performance because several dataset features are correlated.
Support Vector Machine	Achieved the highest accuracy (97.37%), F1 score (0.9630) and MCC (0.9442). Its precision of 100% indicates that it produces no false positive malignant predictions on the test set, while its recall of 92.86% indicates that it correctly identifies most actual malignant cases. Its high AUC (0.9947) also demonstrates excellent class discrimination. This strong performance suggests that SVM can effectively identify complex decision boundaries between the two classes, particularly with feature scaling.


Overall Best-Performing Model
Support Vector Machine (SVM) is the best-performing model overall, achieving the highest accuracy, F1 score and MCC. It provides an excellent balance between correctly identifying malignant tumours and avoiding false positive predictions. However, Logistic Regression achieves the highest AUC, showing that it is also exceptionally effective at distinguishing between the two classes.
Overall Performance Analysis
Overall, all five classification models demonstrated good performance on the Breast Cancer Wisconsin dataset, with noticeable differences in their classification capabilities. SVM emerged as the most effective model, providing the best overall balance between identifying malignant cases and minimising false positive predictions. Logistic Regression also performed exceptionally well, demonstrating strong class discrimination, while KNN showed consistent classification capability. Naive Bayes prioritised precision but was comparatively less effective at identifying all malignant cases, whereas Decision Tree exhibited relatively weaker generalisation. These results suggest that SVM is the most suitable classifier for this dataset, with Logistic Regression serving as a strong alternative.
e. Streamlit Application
Live Streamlit App: Add the deployed Streamlit application URL here once deployment is complete.

The application is intended to allow users to upload test data, select one of the trained classifiers, and view the evaluation metrics and a confusion matrix or classification report. The saved models are loaded for evaluation; the uploaded test data is not used to retrain them.
