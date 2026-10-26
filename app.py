import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

st.title("Breast Cancer Classification Dashboard")

st.write(
    "Compare performance of five machine learning models."
)

uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

selected_model = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ]
)

model_paths = {
    "Logistic Regression":"model/logistic.pkl",
    "Decision Tree":"model/decision_tree.pkl",
    "KNN":"model/knn.pkl",
    "Naive Bayes":"model/naive_bayes.pkl",
    "SVM":"model/svm.pkl"
}

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.write(df.head())

    X = df.drop("target", axis=1)

    y = df["target"]

    scaler = joblib.load(
        "model/scaler.pkl"
    )

    X = scaler.transform(X)

    model = joblib.load(
        model_paths[selected_model]
    )

    y_pred = model.predict(X)

    y_prob = model.predict_proba(X)[:,1]

    accuracy = accuracy_score(
        y,
        y_pred
    )

    auc = roc_auc_score(
        y,
        y_prob
    )

    precision = precision_score(
        y,
        y_pred
    )

    recall = recall_score(
        y,
        y_pred
    )

    f1 = f1_score(
        y,
        y_pred
    )

    mcc = matthews_corrcoef(
        y,
        y_pred
    )

    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", round(accuracy,4))
    st.write("AUC:", round(auc,4))
    st.write("Precision:", round(precision,4))
    st.write("Recall:", round(recall,4))
    st.write("F1 Score:", round(f1,4))
    st.write("MCC:", round(mcc,4))

    st.subheader("Classification Report")

    report = classification_report(
        y,
        y_pred
    )

    st.text(report)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y,
        y_pred
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")

    ax.set_ylabel("Actual")

    st.pyplot(fig)