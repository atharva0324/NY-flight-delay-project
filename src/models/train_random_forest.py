import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

    target_col = "ArrDel15"

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    numeric_features = [
        "Day",
        "Month",
        "DayOfWeek",
        "IsWeekend",
        "IsFixedHoliday",
        "IsHolidayWindow",
        "DepHour",
        "temp",
        "prcp",
        "rain",
        "snow",
        "wspd",
        "airline_delay_rate",
        "origin_delay_rate",
        "Route_Arr_Delay_Rate",
    ]

    categorical_features = [
        "Reporting_Airline",
        "Origin",
        "Dest",
        "Route",
        "DepTimeCategory",
        "Rain_type",
        "Snow_type",
        "Wind_type",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_split=20,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Random Forest Results ===")
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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/models/rf_confusion_matrix.png", dpi=200)
    plt.close()

    metrics_df = pd.DataFrame(
        [
            {
                "model": "RandomForest",
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
            }
        ]
    )
    metrics_df.to_csv("outputs/models/rf_metrics.csv", index=False)

    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # feature importance
    preprocessor_fitted = model.named_steps["preprocessor"]
    classifier_fitted = model.named_steps["classifier"]

    feature_names = preprocessor_fitted.get_feature_names_out()
    importances = classifier_fitted.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    importance_df.to_csv("outputs/models/rf_feature_importance.csv", index=False)

    top_features = importance_df.head(20).sort_values("importance")
    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Random Forest Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("outputs/models/rf_feature_importance.png", dpi=200)
    plt.close()

    print("\nSaved:")
    print("- outputs/models/rf_confusion_matrix.png")
    print("- outputs/models/rf_metrics.csv")
    print("- outputs/models/rf_feature_importance.csv")
    print("- outputs/models/rf_feature_importance.png")
    print("- models/rf_model.pkl")


if __name__ == "__main__":
    main()
    