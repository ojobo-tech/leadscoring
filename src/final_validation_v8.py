from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROCESSED_FILE = Path("data/processed/lead_model_ready_v6.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/final_validation_v8")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "lead_is_converted"

RAW_FEATURES = [
    "status",
    "lead_source",
    "lead_owner_id",
    "owner_role",
    "task_count",
    "open_task_count",
    "closed_task_count",
    "high_priority_task_count",
    "lead_age_days",
    "owner_tenure_days",
    "days_since_last_task_created",
    "days_since_last_task_activity",
    "open_task_ratio",
    "closed_task_ratio",
    "high_priority_task_ratio",
    "task_velocity",
    "high_priority_velocity",
    "task_count_per_owner_day_clean",
    "task_activity_recency_score",
    "task_created_recency_score",
]

CONTEXT_SPECS = [
    ("status", "status_context_raw"),
    ("lead_source", "lead_source_context_raw"),
    ("owner_role", "owner_role_context_raw"),
]

SEGMENT_COLS = ["status", "lead_source", "owner_role"]

SCORE_FEATURE_SPECS = {
    "engagement_score": [
        ("task_count", 0.25, False),
        ("high_priority_task_count", 0.20, False),
        ("open_task_count", 0.15, False),
        ("closed_task_count", 0.15, False),
        ("task_activity_recency_score", 0.15, False),
        ("task_created_recency_score", 0.10, False),
    ],
    "urgency_score": [
        ("days_since_last_task_activity", 0.45, True),
        ("days_since_last_task_created", 0.35, True),
        ("lead_age_days", 0.20, True),
    ],
    "activity_strength_score": [
        ("task_velocity", 0.40, False),
        ("high_priority_velocity", 0.25, False),
        ("task_count_per_owner_day_clean", 0.20, False),
        ("high_priority_task_ratio", 0.15, False),
    ],
}


def load_data() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PROCESSED_FILE}")
    return pd.read_csv(PROCESSED_FILE)


def prepare_base_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found.")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    numeric_cols = [
        "task_count",
        "open_task_count",
        "closed_task_count",
        "high_priority_task_count",
        "lead_age_days",
        "owner_tenure_days",
        "days_since_last_task_created",
        "days_since_last_task_activity",
        "open_task_ratio",
        "closed_task_ratio",
        "high_priority_task_ratio",
        "task_velocity",
        "high_priority_velocity",
        "task_count_per_owner_day_clean",
        "task_activity_recency_score",
        "task_created_recency_score",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)

    if "task_count_per_owner_day_clean" not in df.columns:
        if "task_count" in df.columns and "owner_tenure_days" in df.columns:
            df["task_count_per_owner_day_clean"] = df["task_count"] / (df["owner_tenure_days"] + 1)
        else:
            df["task_count_per_owner_day_clean"] = 0.0

    return df


def fit_smoothed_target_encoding(
    train_df: pd.DataFrame,
    col: str,
    target_col: str = TARGET_COL,
    m: float = 50.0,
) -> Tuple[Dict[str, float], float]:
    overall_rate = train_df[target_col].mean()

    stats = (
        train_df.groupby(col)[target_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "group_mean", "count": "group_count"})
    )

    stats["smoothed_rate"] = (
        (stats["group_count"] * stats["group_mean"]) + (m * overall_rate)
    ) / (stats["group_count"] + m)

    return stats["smoothed_rate"].to_dict(), overall_rate


def apply_smoothed_encoding(
    df: pd.DataFrame,
    col: str,
    mapping: Dict[str, float],
    fallback: float,
) -> pd.Series:
    if col not in df.columns:
        return pd.Series([fallback] * len(df), index=df.index)
    return df[col].map(mapping).fillna(fallback)


def fit_percentile_reference(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return np.array([0.0])
    return np.sort(values)


def apply_percentile_reference(ref: np.ndarray, series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy()
    if len(ref) == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)

    fill_value = ref[0]
    values = np.where(np.isnan(values), fill_value, values)
    pct = np.searchsorted(ref, values, side="right") / len(ref)
    return pd.Series(pct, index=series.index).clip(0, 1)


def fit_validation_state(train_df: pd.DataFrame) -> dict:
    state = {
        "context_mappings": {},
        "context_percentile_refs": {},
        "feature_percentile_refs": {},
        "priority_ref": None,
    }

    train_ctx = train_df.copy()

    # Fit smoothed context encodings on train only
    for source_col, raw_col in CONTEXT_SPECS:
        mapping, fallback = fit_smoothed_target_encoding(train_df, source_col)
        state["context_mappings"][source_col] = {
            "raw_col": raw_col,
            "mapping": mapping,
            "fallback": fallback,
        }
        train_ctx[raw_col] = apply_smoothed_encoding(train_ctx, source_col, mapping, fallback)

    # Fit percentile refs for context raw columns on train only
    for _, raw_col in CONTEXT_SPECS:
        state["context_percentile_refs"][raw_col] = fit_percentile_reference(train_ctx[raw_col])

    # Fit percentile refs for feature columns on train only
    for score_name, specs in SCORE_FEATURE_SPECS.items():
        for col, _, _ in specs:
            if col not in state["feature_percentile_refs"]:
                if col in train_df.columns:
                    state["feature_percentile_refs"][col] = fit_percentile_reference(train_df[col])
                else:
                    state["feature_percentile_refs"][col] = np.array([0.0])

    # Build train scorecard core so we can fit the final priority percentile
    train_scored_core = score_core(train_df.copy(), state)
    state["priority_ref"] = fit_percentile_reference(train_scored_core["priority_score"])

    return state


def score_core(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    df = df.copy()

    # Add raw context columns
    for source_col, info in state["context_mappings"].items():
        raw_col = info["raw_col"]
        mapping = info["mapping"]
        fallback = info["fallback"]
        df[raw_col] = apply_smoothed_encoding(df, source_col, mapping, fallback)

    # Convert raw context columns into percentiles based on train reference
    for _, raw_col in CONTEXT_SPECS:
        ref = state["context_percentile_refs"][raw_col]
        df[f"{raw_col.replace('_raw', '')}_score"] = apply_percentile_reference(ref, df[raw_col])

    # Build context score
    df["context_score"] = (
        0.45 * df["status_context_score"]
        + 0.35 * df["lead_source_context_score"]
        + 0.20 * df["owner_role_context_score"]
    )

    # Build engagement / urgency / activity scores from train-based percentile refs
    for score_name, specs in SCORE_FEATURE_SPECS.items():
        components = []
        for col, weight, inverse in specs:
            ref = state["feature_percentile_refs"].get(col, np.array([0.0]))
            transformed = apply_percentile_reference(ref, df[col] if col in df.columns else pd.Series([0.0] * len(df), index=df.index))
            if inverse:
                transformed = 1 - transformed
            components.append(weight * transformed)
        df[score_name] = sum(components)

    # Final priority score
    df["priority_score"] = 100 * (
        0.40 * df["context_score"]
        + 0.30 * df["urgency_score"]
        + 0.20 * df["engagement_score"]
        + 0.10 * df["activity_strength_score"]
    )

    return df


def score_frame(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    df = score_core(df, state)

    # Priority percentile / bucket based on train distribution
    priority_ref = state["priority_ref"] if state["priority_ref"] is not None else fit_percentile_reference(df["priority_score"])
    df["priority_percentile"] = apply_percentile_reference(priority_ref, df["priority_score"])

    df["priority_bucket"] = pd.cut(
        df["priority_percentile"],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["Cold", "Cool", "Warm", "Hot"],
        include_lowest=True,
    ).astype(str)

    def action_from_bucket(bucket: str) -> str:
        if bucket == "Hot":
            return "Contact today"
        if bucket == "Warm":
            return "Follow up within 48 hours"
        if bucket == "Cool":
            return "Nurture and monitor"
        return "Add to nurture campaign"

    df["recommended_action"] = df["priority_bucket"].apply(action_from_bucket)

    return df


def top_k_metrics(y_true: pd.Series, y_score: pd.Series, k_pct: float = 0.10) -> dict:
    frame = pd.DataFrame(
        {
            "y_true": y_true.reset_index(drop=True),
            "y_score": y_score.reset_index(drop=True),
        }
    ).sort_values("y_score", ascending=False).reset_index(drop=True)

    top_n = max(int(len(frame) * k_pct), 1)
    top_slice = frame.head(top_n)

    overall_rate = frame["y_true"].mean()
    top_rate = top_slice["y_true"].mean()
    lift = top_rate / overall_rate if overall_rate > 0 else np.nan
    capture = top_slice["y_true"].sum() / frame["y_true"].sum() if frame["y_true"].sum() > 0 else np.nan

    return {
        "top_rate": top_rate,
        "lift": lift,
        "capture": capture,
    }


def evaluate_scores(y_true: pd.Series, y_score: pd.Series, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "top_10": top_k_metrics(y_true, y_score, 0.10),
        "top_20": top_k_metrics(y_true, y_score, 0.20),
    }


def build_logistic_benchmark(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[Pipeline, pd.Series]:
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    logistic_prob = pd.Series(
        pipeline.predict_proba(X_test)[:, 1],
        index=X_test.index,
        name="logistic_prob",
    )
    return pipeline, logistic_prob


def segment_stability_report(
    test_df: pd.DataFrame,
    score_cols: List[Tuple[str, str]],
) -> pd.DataFrame:
    rows = []

    for segment_col in SEGMENT_COLS:
        if segment_col not in test_df.columns:
            continue

        for segment_value, seg_df in test_df.groupby(segment_col):
            if len(seg_df) < 20:
                continue

            overall_rate = seg_df[TARGET_COL].mean()

            for method_name, score_col in score_cols:
                sorted_seg = seg_df.sort_values(score_col, ascending=False).reset_index(drop=True)
                top_n_10 = max(int(len(sorted_seg) * 0.10), 1)
                top_n_20 = max(int(len(sorted_seg) * 0.20), 1)

                top10 = sorted_seg.head(top_n_10)
                top20 = sorted_seg.head(top_n_20)

                top10_rate = top10[TARGET_COL].mean()
                top20_rate = top20[TARGET_COL].mean()

                rows.append(
                    {
                        "segment": segment_col,
                        "segment_value": segment_value,
                        "method": method_name,
                        "n": len(seg_df),
                        "overall_rate": overall_rate,
                        "top10_rate": top10_rate,
                        "top10_lift": top10_rate / overall_rate if overall_rate > 0 else np.nan,
                        "top20_rate": top20_rate,
                        "top20_lift": top20_rate / overall_rate if overall_rate > 0 else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    df = prepare_base_df(load_data())

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COL],
    )

    # Fit scorecard state on train only
    state = fit_validation_state(train_df.copy())

    # Score train and test using the train-fitted state
    train_scored = score_frame(train_df.copy(), state)
    test_scored = score_frame(test_df.copy(), state)

    # Scorecard validation
    scorecard_threshold = train_scored["priority_score"].median()
    scorecard_metrics = evaluate_scores(
        test_scored[TARGET_COL],
        test_scored["priority_score"],
        threshold=scorecard_threshold,
    )

    # Logistic benchmark on the same raw features, excluding scorecard outputs
    feature_cols = RAW_FEATURES.copy()
    logistic_model, logistic_prob = build_logistic_benchmark(train_df, test_df, feature_cols)

    logistic_threshold = 0.50
    logistic_metrics = evaluate_scores(
        test_scored[TARGET_COL],
        logistic_prob,
        threshold=logistic_threshold,
    )

    comparison = pd.DataFrame(
        [
            {
                "method": "scorecard_v8",
                "accuracy": scorecard_metrics["accuracy"],
                "precision": scorecard_metrics["precision"],
                "recall": scorecard_metrics["recall"],
                "f1": scorecard_metrics["f1"],
                "roc_auc": scorecard_metrics["roc_auc"],
                "top_10_rate": scorecard_metrics["top_10"]["top_rate"],
                "top_10_lift": scorecard_metrics["top_10"]["lift"],
                "top_10_capture": scorecard_metrics["top_10"]["capture"],
                "top_20_rate": scorecard_metrics["top_20"]["top_rate"],
                "top_20_lift": scorecard_metrics["top_20"]["lift"],
                "top_20_capture": scorecard_metrics["top_20"]["capture"],
            },
            {
                "method": "logistic_regression",
                "accuracy": logistic_metrics["accuracy"],
                "precision": logistic_metrics["precision"],
                "recall": logistic_metrics["recall"],
                "f1": logistic_metrics["f1"],
                "roc_auc": logistic_metrics["roc_auc"],
                "top_10_rate": logistic_metrics["top_10"]["top_rate"],
                "top_10_lift": logistic_metrics["top_10"]["lift"],
                "top_10_capture": logistic_metrics["top_10"]["capture"],
                "top_20_rate": logistic_metrics["top_20"]["top_rate"],
                "top_20_lift": logistic_metrics["top_20"]["lift"],
                "top_20_capture": logistic_metrics["top_20"]["capture"],
            },
        ]
    ).sort_values(by=["top_10_lift", "roc_auc", "f1"], ascending=False)

    segment_report = segment_stability_report(
        test_scored.assign(logistic_prob=logistic_prob),
        score_cols=[
            ("scorecard_v8", "priority_score"),
            ("logistic_regression", "logistic_prob"),
        ],
    )

    # Save outputs
    comparison_file = REPORTS_DIR / "overall_comparison.csv"
    segment_file = REPORTS_DIR / "segment_stability.csv"
    test_pred_file = REPORTS_DIR / "test_predictions.csv"

    comparison.to_csv(comparison_file, index=False)
    segment_report.to_csv(segment_file, index=False)

    test_out = test_scored.copy()
    test_out["logistic_prob"] = logistic_prob
    test_out.to_csv(test_pred_file, index=False)

    # Save logistic benchmark model
    joblib.dump(logistic_model, MODELS_DIR / "logistic_benchmark_v8.joblib")

    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)
    print(comparison)

    print(f"\nSaved overall comparison to: {comparison_file}")
    print(f"Saved segment stability report to: {segment_file}")
    print(f"Saved test predictions to: {test_pred_file}")
    print(f"Saved logistic benchmark model to: {MODELS_DIR / 'logistic_benchmark_v8.joblib'}")


if __name__ == "__main__":
    main()