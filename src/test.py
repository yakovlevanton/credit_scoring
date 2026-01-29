from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.features import (
    CAT_FEATURES,
    build_bureau_agg,
    build_prev_app_agg,
    transform_application,
)


@dataclass(frozen=True)
class TestBundle:
    X_test: pd.DataFrame
    test_ids: pd.Series


def _load_raw_test(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)

    app_test_path = data_dir / "application_test.csv"
    bureau_path = data_dir / "bureau.csv"
    prev_path = data_dir / "previous_application.csv"

    missing = [str(p) for p in [app_test_path, bureau_path, prev_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files in {data_dir}: {missing}")

    app_test = pd.read_csv(app_test_path)
    bureau = pd.read_csv(bureau_path)
    prev = pd.read_csv(prev_path)
    return app_test, bureau, prev


def make_test_features(data_dir: str | Path) -> TestBundle:
    app_test, bureau, prev = _load_raw_test(data_dir)

    test_ids = app_test["SK_ID_CURR"]

    X_app = transform_application(app_test)
    X_app = X_app.copy()
    X_app["SK_ID_CURR"] = test_ids.values

    bureau_agg = build_bureau_agg(bureau)
    prev_agg = build_prev_app_agg(prev)

    X_test = X_app.merge(bureau_agg, on="SK_ID_CURR", how="left")
    X_test = X_test.merge(prev_agg, on="SK_ID_CURR", how="left")
    X_test = X_test.drop(columns=["SK_ID_CURR"], errors="ignore")

    return TestBundle(X_test=X_test, test_ids=test_ids)


def load_model(model_path: str | Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def predict_proba(model: CatBoostClassifier, X_test: pd.DataFrame) -> pd.Series:
    X = X_test.copy()
    for col in CAT_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("object").fillna("__MISSING__")

    cat_cols: List[str] = [c for c in CAT_FEATURES if c in X.columns]
    pool = Pool(X, cat_features=cat_cols)
    proba = model.predict_proba(pool)[:, 1]
    return pd.Series(proba)


def make_submission(test_ids: pd.Series, proba: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"SK_ID_CURR": test_ids.values, "TARGET": proba.values})


def predict_and_save(
    data_dir: str | Path,
    model_path: str | Path,
    out_csv: str | Path,
) -> None:
    bundle = make_test_features(data_dir)
    model = load_model(model_path)

    proba = predict_proba(model, bundle.X_test)
    sub = make_submission(bundle.test_ids, proba)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)

