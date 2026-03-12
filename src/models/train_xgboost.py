import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
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

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

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
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== XGBoost Results ===")
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
    plt.title("XGBoost Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/models/xgb_confusion_matrix.png", dpi=200)
    plt.close()

    metrics_df = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
            }
        ]
    )
    metrics_df.to_csv("outputs/models/xgb_metrics.csv", index=False)

    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)

    preprocessor_fitted = model.named_steps["preprocessor"]
    classifier_fitted = model.named_steps["classifier"]

    feature_names = preprocessor_fitted.get_feature_names_out()
    importances = classifier_fitted.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    importance_df.to_csv("outputs/models/xgb_feature_importance.csv", index=False)

    top_features = importance_df.head(20).sort_values("importance")
    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("XGBoost Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("outputs/models/xgb_feature_importance.png", dpi=200)
    plt.close()

    print("\nSaved:")
    print("- outputs/models/xgb_confusion_matrix.png")
    print("- outputs/models/xgb_metrics.csv")
    print("- outputs/models/xgb_feature_importance.csv")
    print("- outputs/models/xgb_feature_importance.png")
    print("- models/xgb_model.pkl")


if __name__ == "__main__":
    main()