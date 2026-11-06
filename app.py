
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Breast Cancer Classification")
st.write(
    "Machine Learning application to classify tumours "
    "as Benign or Malignant using five ML classifiers."
)

st.divider()


# --------------------------------------------------
# 2. DEFINE FEATURE NAMES
# --------------------------------------------------

FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean',
    'area_mean', 'smoothness_mean', 'compactness_mean',
    'concavity_mean', 'concave_points_mean', 'symmetry_mean',
    'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave_points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]


# --------------------------------------------------
# 3. DEFINE SAVED MODEL LOCATIONS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}


# --------------------------------------------------
# 4. LOAD SAVED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# 5. ENCODE TARGET VARIABLE
# --------------------------------------------------

def encode_target(target):

    if pd.api.types.is_numeric_dtype(target):

        encoded = pd.to_numeric(target, errors="coerce")

        if encoded.isna().any() or not set(encoded.unique()).issubset({0, 1}):
            raise ValueError("Diagnosis must contain only 0 and 1.")

        return encoded.astype(int)

    mapping = {
        "b": 0,
        "benign": 0,
        "0": 0,
        "0.0": 0,
        "m": 1,
        "malignant": 1,
        "1": 1,
        "1.0": 1
    }

    encoded = target.astype(str).str.strip().str.lower().map(mapping)

    if encoded.isna().any():
        raise ValueError(
            "Diagnosis must contain B, M, Benign, Malignant, 0 or 1."
        )

    return encoded.astype(int)


# --------------------------------------------------
# 6. MODEL SELECTION
# --------------------------------------------------

st.subheader("Select Classification Model")

selected_model = st.selectbox(
    "Choose a machine learning algorithm:",
    list(MODEL_FILES.keys())
)

st.divider()


# --------------------------------------------------
# 7. UPLOAD TEST DATA
# --------------------------------------------------

st.subheader("Upload Test Dataset")

st.write(
    "Upload a CSV file containing all 30 input features "
    "and the diagnosis column."
)

uploaded_file = st.file_uploader(
    "Choose your test CSV file",
    type=["csv"]
)

# Provide the existing test CSV as an example

sample_path = BASE_DIR / "test_data.csv"

if sample_path.exists():

    st.download_button(
        label="Download Example Test Dataset",
        data=sample_path.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 8. PROCESS UPLOADED DATA
# --------------------------------------------------

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        # Standardise column names
        data.columns = data.columns.str.strip().str.lower()

        st.write("### Uploaded Dataset Preview")
        st.dataframe(data.head())

        # Check required features
        missing_features = [
            feature for feature in FEATURES
            if feature not in data.columns
        ]

        if missing_features:

            st.error(
                f"Missing required features: {missing_features}"
            )
            st.stop()

        # Check target column
        if "diagnosis" not in data.columns:

            st.error(
                "The uploaded CSV must contain a diagnosis column."
            )
            st.stop()

        # Extract input features
        X_test = data[FEATURES].apply(
            pd.to_numeric, errors="coerce"
        )

        if X_test.isnull().any().any():

            st.error(
                "The input features contain missing or non-numeric values."
            )
            st.stop()

        # Extract and encode actual target
        y_test = encode_target(data["diagnosis"])

        if len(y_test.unique()) < 2:

            st.warning(
                "The uploaded dataset contains only one diagnosis class. "
                "AUC cannot be calculated."
            )

        # --------------------------------------------------
        # 9. LOAD SELECTED MODEL
        # --------------------------------------------------

        model_path = MODEL_DIR / MODEL_FILES[selected_model]

        if not model_path.exists():

            st.error(
                f"Model file not found: {model_path}"
            )
            st.stop()

        with st.spinner("Loading saved model..."):

            model = load_model(str(model_path))

        st.success(f"{selected_model} loaded successfully!")


        # --------------------------------------------------
        # 10. MAKE PREDICTIONS
        # --------------------------------------------------

        y_pred = model.predict(X_test)


        # --------------------------------------------------
        # 11. CALCULATE EVALUATION METRICS
        # --------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test, y_pred, zero_division=0
        )

        recall = recall_score(
            y_test, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        mcc = matthews_corrcoef(y_test, y_pred)

        # Calculate AUC using malignant-class probabilities
        if len(y_test.unique()) == 2:

            if not hasattr(model, "predict_proba"):

                st.error("The selected model does not support probability predictions.")
                st.stop()

            probabilities = model.predict_proba(X_test)

            positive_class_index = list(model.classes_).index(1)

            auc = roc_auc_score(
                y_test,
                probabilities[:, positive_class_index]
            )

        else:
            auc = None


        # --------------------------------------------------
        # 12. DISPLAY EVALUATION METRICS
        # --------------------------------------------------

        st.divider()

        st.subheader(f"Performance Metrics: {selected_model}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}" if auc is not None else "N/A")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")


        # --------------------------------------------------
        # 13. CONFUSION MATRIX
        # --------------------------------------------------

        st.divider()

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benign", "Malignant"]
        )

        display.plot(
            ax=ax,
            values_format="d",
            colorbar=False
        )

        ax.set_title("Actual vs Predicted Diagnosis")

        st.pyplot(fig)
        plt.close(fig)


        # --------------------------------------------------
        # 14. CLASSIFICATION REPORT
        # --------------------------------------------------

        st.divider()

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df.round(4))


        # --------------------------------------------------
        # 15. INDIVIDUAL PREDICTIONS
        # --------------------------------------------------

        st.divider()

        st.subheader("Prediction Results")

        results = pd.DataFrame({
            "Actual Diagnosis": y_test.map({
                0: "Benign",
                1: "Malignant"
            }).values,

            "Predicted Diagnosis": pd.Series(y_pred).map({
                0: "Benign",
                1: "Malignant"
            }).values
        })

        st.dataframe(results)

        st.success("Model evaluation completed successfully!")


    except Exception as e:

        st.error(f"An error occurred: {e}")

else:

    st.info(
        "Please upload a test CSV file to begin model evaluation."
    )
