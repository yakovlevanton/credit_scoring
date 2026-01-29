from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from src.features import (
    CAT_FEATURES,
    build_bureau_agg,
    build_prev_app_agg,
    transform_application,
)


@dataclass(frozen=True)
class TrainBundle:
    X: pd.DataFrame
    y: pd.Series


def _load_raw_train(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)

    app_train_path = data_dir / "application_train.csv"
    bureau_path = data_dir / "bureau.csv"
    prev_path = data_dir / "previous_application.csv"

    missing = [str(p) for p in [app_train_path, bureau_path, prev_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files in {data_dir}: {missing}")

    app_train = pd.read_csv(app_train_path)
    bureau = pd.read_csv(bureau_path)
    prev = pd.read_csv(prev_path)
    return app_train, bureau, prev


def make_train_features(data_dir: str | Path) -> TrainBundle:
    app_train, bureau, prev = _load_raw_train(data_dir)

    if "TARGET" not in app_train.columns:
        raise ValueError("application_train.csv must contain TARGET")

    train_ids = app_train["SK_ID_CURR"]
    y = app_train["TARGET"]

    X_app = transform_application(app_train.drop(columns=["TARGET"], errors="ignore"))
    X_app = X_app.copy()
    X_app["SK_ID_CURR"] = train_ids.values

    bureau_agg = build_bureau_agg(bureau)
    prev_agg = build_prev_app_agg(prev)

    X = X_app.merge(bureau_agg, on="SK_ID_CURR", how="left")
    X = X.merge(prev_agg, on="SK_ID_CURR", how="left")
    X = X.drop(columns=["SK_ID_CURR"], errors="ignore")

    return TrainBundle(X=X, y=y)


def train_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
) -> CatBoostClassifier:

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    for col in CAT_FEATURES:
        if col in X_tr.columns:
            X_tr[col] = X_tr[col].astype("object").fillna("__MISSING__")
            X_val[col] = X_val[col].astype("object").fillna("__MISSING__")

    cat_cols: List[str] = [c for c in CAT_FEATURES if c in X.columns]
    train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        max_depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        early_stopping_rounds=200,
        verbose=200,
        random_seed=random_state,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


def train_and_save(
    data_dir: str | Path,
    model_path: str | Path,
    *,
    random_state: int = 42,
) -> None:
    bundle = make_train_features(data_dir)
    model = train_catboost(bundle.X, bundle.y, random_state=random_state)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

