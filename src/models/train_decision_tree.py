import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.data_preparation import load_split_data


def main():
    train_df, test_df = load_split_data()

    X_train = train_df.drop(columns=["ArrDel15"])
    y_train = train_df["ArrDel15"]

    X_test = test_df.drop(columns=["ArrDel15"])
    y_test = test_df["ArrDel15"]

    numeric_features = [
        "Day", "Month", "DayOfWeek", "IsWeekend",
        "IsFixedHoliday", "IsHolidayWindow", "DepHour",
        "temp", "prcp", "rain", "snow", "wspd",
        "airline_delay_rate", "origin_delay_rate", "Route_Arr_Delay_Rate"
    ]

    categorical_features = [
        "Reporting_Airline", "Origin", "Dest", "Route",
        "DepTimeCategory", "Rain_type", "Snow_type", "Wind_type"
    ]

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Decision Tree Results ===")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall   :", round(recall_score(y_test, y_pred), 4))
    print("F1 Score :", round(f1_score(y_test, y_pred), 4))
    print("ROC AUC  :", round(roc_auc_score(y_test, y_prob), 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title("Decision Tree Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/models/dt_confusion_matrix.png", dpi=200)
    plt.close()

    pd.DataFrame([{
        "model": "DecisionTree",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }]).to_csv("outputs/models/dt_metrics.csv", index=False)

    with open("models/dt_model.pkl", "wb") as f:
        pickle.dump(model, f)

    preprocessor_fitted = model.named_steps["preprocessor"]
    classifier_fitted = model.named_steps["classifier"]

    feature_names = preprocessor_fitted.get_feature_names_out()
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier_fitted.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df.to_csv("outputs/models/dt_feature_importance.csv", index=False)

    top_features = importance_df.head(20).sort_values("importance")
    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Decision Tree Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("outputs/models/dt_feature_importance.png", dpi=200)
    plt.close()

    print("\nSaved:")
    print("- outputs/models/dt_confusion_matrix.png")
    print("- outputs/models/dt_metrics.csv")
    print("- outputs/models/dt_feature_importance.csv")
    print("- outputs/models/dt_feature_importance.png")
    print("- models/dt_model.pkl")


if __name__ == "__main__":
    main()