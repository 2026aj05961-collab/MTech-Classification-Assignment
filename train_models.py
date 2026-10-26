import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

import joblib
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

df = pd.read_csv(
    "breast_cancer.csv"
)

X = df.drop(
    "target",
    axis=1
)

y = df["target"]

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
X_test_original = X_test.copy()

#scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_test_original["target"] = y_test.values

X_test_original.to_csv(
    "test_data.csv",
    index=False
)

#creating model folder
os.makedirs(
    "model",
    exist_ok=True
)

#training logistic regression
lr = LogisticRegression(
    max_iter=10000
)

lr.fit(
    X_train,
    y_train
)

#training decision tree
dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(
    X_train,
    y_train
)

#training KNN
knn = KNeighborsClassifier()

knn.fit(
    X_train,
    y_train
)

#training Naive Bayes
nb = GaussianNB()

nb.fit(
    X_train,
    y_train
)

#training SVM
svm = SVC(
    probability=True
)

svm.fit(
    X_train,
    y_train
)

#saving models
joblib.dump(
    lr,
    "model/logistic.pkl"
)

joblib.dump(
    dt,
    "model/decision_tree.pkl"
)

joblib.dump(
    knn,
    "model/knn.pkl"
)

joblib.dump(
    nb,
    "model/naive_bayes.pkl"
)

joblib.dump(
    svm,
    "model/svm.pkl"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)
#evalting the model
def evaluate_model(model, model_name):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    print("\n")
    print(model_name)

    print("Accuracy :", accuracy)
    print("AUC :", auc)
    print("Precision :", precision)
    print("Recall :", recall)
    print("F1 :", f1)
    print("MCC :", mcc)

    return [
        model_name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ]

#creating comparison table
results = []

results.append(
    evaluate_model(
        lr,
        "Logistic Regression"
    )
)

results.append(
    evaluate_model(
        dt,
        "Decision Tree"
    )
)

results.append(
    evaluate_model(
        knn,
        "KNN"
    )
)

results.append(
    evaluate_model(
        nb,
        "Naive Bayes"
    )
)

results.append(
    evaluate_model(
        svm,
        "SVM"
    )
)
#saving results in csv
comparison_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
)

comparison_df.to_csv(
    "results.csv",
    index=False
)

print("\n")
print(comparison_df)

# create test dataframe using ORIGINAL column names

X_test_df = pd.DataFrame(
    X_test,
    columns=X.columns
)

X_test_df["target"] = y_test.values

X_test_df.to_csv(
    "test_data.csv",
    index=False
)

