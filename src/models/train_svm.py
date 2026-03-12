import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.models.data_preparation import load_split_data


def main():

    train_df, test_df = load_split_data()

    X_train = train_df.drop(columns=["ArrDel15"])
    y_train = train_df["ArrDel15"]

    X_test = test_df.drop(columns=["ArrDel15"])
    y_test = test_df["ArrDel15"]

    numeric_features = [
        "Day","Month","DayOfWeek","IsWeekend",
        "IsFixedHoliday","IsHolidayWindow","DepHour",
        "temp","prcp","rain","snow","wspd",
        "airline_delay_rate","origin_delay_rate","Route_Arr_Delay_Rate"
    ]

    categorical_features = [
        "Reporting_Airline","Origin","Dest","Route",
        "DepTimeCategory","Rain_type","Snow_type","Wind_type"
    ]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric_features),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    base_model = LinearSVC(class_weight="balanced")

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", CalibratedClassifierCV(base_model))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print("\n=== SVM Results ===")
    print("Accuracy :", round(accuracy_score(y_test,y_pred),4))
    print("Precision:", round(precision_score(y_test,y_pred),4))
    print("Recall   :", round(recall_score(y_test,y_pred),4))
    print("F1 Score :", round(f1_score(y_test,y_pred),4))
    print("ROC AUC  :", round(roc_auc_score(y_test,y_prob),4))

    print("\nClassification Report\n")
    print(classification_report(y_test,y_pred))

    os.makedirs("outputs/models",exist_ok=True)
    os.makedirs("models",exist_ok=True)

    cm = confusion_matrix(y_test,y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title("SVM Confusion Matrix")
    plt.savefig("outputs/models/svm_confusion_matrix.png")
    plt.close()

    with open("models/svm_model.pkl","wb") as f:
        pickle.dump(model,f)


if __name__ == "__main__":
    main()