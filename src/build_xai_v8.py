from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline


PROCESSED_FILE = Path("reports/final_validation_v8/test_predictions.csv")
MODEL_FILE = Path("models/logistic_benchmark_v8.joblib")
OUTPUT_DIR = Path("reports/xai_v8")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load_inputs() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PROCESSED_FILE}")
    return pd.read_csv(PROCESSED_FILE)


def load_model() -> Pipeline:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    if not hasattr(model, "named_steps"):
        raise TypeError("Loaded artifact is not a sklearn Pipeline.")
    return model


def select_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in RAW_FEATURES if c in df.columns]
    if not feature_cols:
        raise KeyError("No model input features were found in the input file.")
    return df[feature_cols].copy()


def get_feature_names(pipeline: Pipeline) -> List[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def transform_features(pipeline: Pipeline, X: pd.DataFrame):
    preprocessor = pipeline.named_steps["preprocessor"]
    Xt = preprocessor.transform(X)
    return Xt


def dense_matrix(Xt):
    if sparse.issparse(Xt):
        return Xt.toarray()
    return np.asarray(Xt)


def get_logistic_weights(pipeline: Pipeline) -> Tuple[np.ndarray, float]:
    model = pipeline.named_steps["model"]
    coef = model.coef_.ravel()
    intercept = float(model.intercept_[0])
    return coef, intercept


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def friendly_feature_name(feature_name: str) -> str:
    name = feature_name.replace("num__", "").replace("cat__", "")

    if "lead_source_" in name:
        return "lead source"
    if "status_" in name:
        return "status"
    if "owner_role_" in name:
        return "owner role"
    if "lead_owner_id_" in name:
        return "assigned owner"
    if name == "task_count":
        return "task count"
    if name == "open_task_count":
        return "open task count"
    if name == "closed_task_count":
        return "closed task count"
    if name == "high_priority_task_count":
        return "high priority task count"
    if name == "lead_age_days":
        return "lead age"
    if name == "owner_tenure_days":
        return "owner tenure"
    if name == "days_since_last_task_created":
        return "days since last task created"
    if name == "days_since_last_task_activity":
        return "days since last task activity"
    if name == "open_task_ratio":
        return "open task ratio"
    if name == "closed_task_ratio":
        return "closed task ratio"
    if name == "high_priority_task_ratio":
        return "high priority task ratio"
    if name == "task_velocity":
        return "task velocity"
    if name == "high_priority_velocity":
        return "high priority velocity"
    if name == "task_count_per_owner_day_clean":
        return "task intensity per owner day"
    if name == "task_activity_recency_score":
        return "task activity recency"
    if name == "task_created_recency_score":
        return "task created recency"

    return name.replace("_", " ")


def is_categorical_onehot(feature_name: str) -> bool:
    return feature_name.startswith("cat__")


def parse_category_label(feature_name: str) -> str:
    name = feature_name.replace("cat__", "")
    if "_" not in name:
        return friendly_feature_name(feature_name)
    base, value = name.split("_", 1)
    if base == "status":
        return f"Status = {value}"
    if base == "lead_source":
        return f"Lead source = {value}"
    if base == "owner_role":
        return f"Owner role = {value}"
    if base == "lead_owner_id":
        return "Specific owner assignment"
    return f"{base.replace('_', ' ').title()} = {value}"


INTUITIVE_HIGER_IS_BETTER = {
    "task_count",
    "open_task_count",
    "closed_task_count",
    "high_priority_task_count",
    "open_task_ratio",
    "closed_task_ratio",
    "high_priority_task_ratio",
    "task_velocity",
    "high_priority_velocity",
    "task_count_per_owner_day_clean",
    "task_activity_recency_score",
    "task_created_recency_score",
}

INTUITIVE_LOWER_IS_BETTER = {
    "lead_age_days",
    "owner_tenure_days",
    "days_since_last_task_created",
    "days_since_last_task_activity",
}


def interpret_numeric_reason(raw_feature: str, value: float, contribution: float) -> str:
    friendly = friendly_feature_name(raw_feature)

    if raw_feature in INTUITIVE_HIGER_IS_BETTER:
        if contribution >= 0:
            return f"Higher {friendly} increased priority"
        return f"Lower {friendly} reduced priority"

    if raw_feature in INTUITIVE_LOWER_IS_BETTER:
        if contribution >= 0:
            return f"Lower {friendly} increased priority"
        return f"Higher {friendly} reduced priority"

    if contribution >= 0:
        return f"{friendly} supported higher priority"
    return f"{friendly} reduced priority"


def build_contribution_frame(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], np.ndarray, np.ndarray]:
    feature_names = get_feature_names(pipeline)
    Xt = transform_features(pipeline, X)
    Xt_dense = dense_matrix(Xt)

    coef, intercept = get_logistic_weights(pipeline)
    contributions = Xt_dense * coef

    logit = contributions.sum(axis=1) + intercept
    prob = sigmoid(logit)

    contrib_df = pd.DataFrame(contributions, columns=feature_names, index=X.index)
    return contrib_df, feature_names, logit, prob


def explain_row(
    raw_row: pd.Series,
    contrib_row: pd.Series,
    top_n: int = 3,
) -> Tuple[List[str], str]:
    reason_rows = []

    for feat_name, contrib_value in contrib_row.sort_values(ascending=False).items():
        raw_feat = feat_name.replace("num__", "").replace("cat__", "")

        if "lead_owner_id" in raw_feat:
            continue

        raw_val = raw_row.get(raw_feat, np.nan)

        if is_categorical_onehot(feat_name):
            if contrib_value <= 0:
                continue
            reason_text = parse_category_label(feat_name)
        else:
            reason_text = interpret_numeric_reason(raw_feat, raw_val, contrib_value)

        reason_rows.append((reason_text, contrib_value))

        if len(reason_rows) >= top_n:
            break

    if not reason_rows:
        reason_rows = [("No strong positive driver detected", 0.0)]

    reasons = [r[0] for r in reason_rows]

    if len(reasons) == 1:
        summary = f"This lead is ranked higher because {reasons[0].lower()}."
    elif len(reasons) == 2:
        summary = f"This lead is ranked higher because {reasons[0].lower()} and {reasons[1].lower()}."
    else:
        summary = (
            f"This lead is ranked higher because {reasons[0].lower()}, "
            f"{reasons[1].lower()}, and {reasons[2].lower()}."
        )

    return reasons, summary


def global_importance(contrib_df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    importance = contrib_df.abs().mean(axis=0).sort_values(ascending=False)
    out = pd.DataFrame(
        {
            "feature": importance.index,
            "mean_abs_contribution": importance.values,
        }
    )

    out["friendly_feature"] = out["feature"].apply(friendly_feature_name)
    out.to_csv(OUTPUT_DIR / "global_feature_importance.csv", index=False)
    return out


def build_xai_outputs(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    X = select_feature_df(df)
    contrib_df, feature_names, logit, prob = build_contribution_frame(pipeline, X)

    out = df.copy()
    out["xai_logit"] = logit
    out["xai_probability"] = prob

    reason_1_list = []
    reason_2_list = []
    reason_3_list = []
    summary_list = []

    for idx in out.index:
        raw_row = X.loc[idx]
        contrib_row = contrib_df.loc[idx]

        reasons, summary = explain_row(raw_row, contrib_row, top_n=3)

        reason_1_list.append(reasons[0] if len(reasons) > 0 else "")
        reason_2_list.append(reasons[1] if len(reasons) > 1 else "")
        reason_3_list.append(reasons[2] if len(reasons) > 2 else "")
        summary_list.append(summary)

    out["reason_1"] = reason_1_list
    out["reason_2"] = reason_2_list
    out["reason_3"] = reason_3_list
    out["xai_summary"] = summary_list

    out.to_csv(OUTPUT_DIR / "lead_level_explanations.csv", index=False)

    return out


def save_local_summary(df: pd.DataFrame) -> None:
    summary_cols = [
        "status",
        "lead_source",
        "owner_role",
        "priority_bucket",
        "recommended_action",
        "lead_is_converted",
        "xai_probability",
        "reason_1",
        "reason_2",
        "reason_3",
        "xai_summary",
    ]
    existing = [c for c in summary_cols if c in df.columns]
    df[existing].to_csv(OUTPUT_DIR / "xai_preview.csv", index=False)


def main() -> None:
    df = load_inputs()
    pipeline = load_model()

    explained_df = build_xai_outputs(df, pipeline)
    save_local_summary(explained_df)

    importance = global_importance(
        pd.DataFrame(
            dense_matrix(transform_features(pipeline, select_feature_df(df)))
            * get_logistic_weights(pipeline)[0],
            columns=get_feature_names(pipeline),
            index=df.index,
        ),
        get_feature_names(pipeline),
    )

    print("=" * 80)
    print("XAI OUTPUTS CREATED")
    print("=" * 80)
    print(f"Saved lead explanations to: {OUTPUT_DIR / 'lead_level_explanations.csv'}")
    print(f"Saved XAI preview to: {OUTPUT_DIR / 'xai_preview.csv'}")
    print(f"Saved global importance to: {OUTPUT_DIR / 'global_feature_importance.csv'}")
    print("\nTop global drivers:")
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()