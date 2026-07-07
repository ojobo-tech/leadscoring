from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        "xgboost is not installed. Install it with: python -m pip install xgboost"
    ) from e

try:
    from catboost import CatBoostClassifier
except ImportError as e:
    raise ImportError(
        "catboost is not installed. Install it with: python -m pip install catboost"
    ) from e


PROCESSED_FILE = Path("data/processed/lead_priority_dataset_v8.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/modeling_v8")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "lead_is_converted"

# Exclude all scorecard outputs and anything derived from the target.
DROP_COLS = [
    "priority_score",
    "priority_percentile",
    "priority_bucket",
    "recommended_action",
    "context_score",
    "engagement_score",
    "urgency_score",
    "activity_strength_score",
    "status_context_score",
    "lead_source_context_score",
    "owner_role_context_score",
]


def load_data() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PROCESSED_FILE}")
    return pd.read_csv(PROCESSED_FILE)


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataset.")

    if df[TARGET_COL].dtype == "bool":
        df[TARGET_COL] = df[TARGET_COL].astype(int)
    else:
        df[TARGET_COL] = (
            df[TARGET_COL]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
        )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_candidates = [c for c in DROP_COLS if c in df.columns]
    feature_df = df.drop(columns=drop_candidates + [TARGET_COL], errors="ignore")
    return feature_df


def split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def get_feature_types(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str], scale_numeric: bool):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


def get_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        ),
        "catboost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            random_seed=42,
            verbose=0,
        ),
    }


def top_k_metrics(y_true, y_prob, k_pct: float = 0.10) -> dict:
    df = pd.DataFrame({"y_true": y_true.reset_index(drop=True), "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)

    top_n = max(int(len(df) * k_pct), 1)
    top_slice = df.head(top_n)

    overall_rate = df["y_true"].mean()
    top_rate = top_slice["y_true"].mean()
    lift = top_rate / overall_rate if overall_rate > 0 else 0.0
    capture = top_slice["y_true"].sum() / df["y_true"].sum() if df["y_true"].sum() > 0 else 0.0

    return {
        "top_pct": k_pct,
        "top_rate": top_rate,
        "lift": lift,
        "capture": capture,
    }


def evaluate_model(model_name: str, pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        top10 = top_k_metrics(y_test, y_prob, 0.10)
        top20 = top_k_metrics(y_test, y_prob, 0.20)
    else:
        y_prob = None
        roc_auc = None
        top10 = {"top_rate": None, "lift": None, "capture": None}
        top20 = {"top_rate": None, "lift": None, "capture": None}

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "top_10_rate": top10["top_rate"],
        "top_10_lift": top10["lift"],
        "top_10_capture": top10["capture"],
        "top_20_rate": top20["top_rate"],
        "top_20_lift": top20["lift"],
        "top_20_capture": top20["capture"],
        "pipeline": pipeline,
    }


def main() -> None:
    df = load_data()
    df = prepare_target(df)

    X = build_features(df)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = split_data(X, y)
    numeric_cols, categorical_cols = get_feature_types(X_train)

    models = get_models()
    results = []

    for model_name, estimator in models.items():
        print(f"\nTraining {model_name}...")

        scale_numeric = model_name == "logistic_regression"
        preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale_numeric)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        result = evaluate_model(model_name, pipeline, X_test, y_test)
        results.append(result)

        print(
            f"{model_name} | "
            f"acc={result['accuracy']:.4f} | "
            f"prec={result['precision']:.4f} | "
            f"recall={result['recall']:.4f} | "
            f"f1={result['f1']:.4f} | "
            f"roc_auc={result['roc_auc']:.4f} | "
            f"top10_lift={result['top_10_lift']:.4f}"
        )

    results_df = pd.DataFrame(results).drop(columns=["pipeline"])
    results_df = results_df.sort_values(by=["top_10_lift", "roc_auc", "f1"], ascending=False)

    comparison_file = REPORTS_DIR / "model_comparison_v8.csv"
    results_df.to_csv(comparison_file, index=False)

    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = next(r["pipeline"] for r in results if r["model"] == best_model_name)

    best_model_file = MODELS_DIR / "best_model_v8.joblib"
    joblib.dump(best_pipeline, best_model_file)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON V8")
    print("=" * 80)
    print(results_df)

    print(f"\nSaved comparison table to: {comparison_file}")
    print(f"Saved best model to: {best_model_file}")
    print(f"Best model: {best_model_name}")


if __name__ == "__main__":
    main()