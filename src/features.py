from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List
import numpy as np
import pandas as pd


CAT_FEATURES: List[str] = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE",
]


DROP_COLS_STATIC: List[str] = [
    # ID
    "SK_ID_CURR",

    # COMMONAREA
    "COMMONAREA_AVG", "COMMONAREA_MODE", "COMMONAREA_MEDI",

    # NONLIVING APARTMENTS
    "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAPARTMENTS_MEDI",

    # NONLIVING AREA
    "NONLIVINGAREA_AVG", "NONLIVINGAREA_MODE", "NONLIVINGAREA_MEDI",

    # LAND AREA
    "LANDAREA_AVG", "LANDAREA_MODE", "LANDAREA_MEDI",

    # BASEMENT
    "BASEMENTAREA_AVG", "BASEMENTAREA_MODE", "BASEMENTAREA_MEDI",

    # LIVING APARTMENTS
    "LIVINGAPARTMENTS_AVG", "LIVINGAPARTMENTS_MODE", "LIVINGAPARTMENTS_MEDI",

    # FLOORS MIN
    "FLOORSMIN_AVG", "FLOORSMIN_MODE", "FLOORSMIN_MEDI",

    # FLOORS MAX
    "FLOORSMAX_AVG", "FLOORSMAX_MODE", "FLOORSMAX_MEDI",

    # ENTRANCES
    "ENTRANCES_AVG", "ENTRANCES_MODE", "ENTRANCES_MEDI",

    # YEARS
    "YEARS_BUILD_AVG", "YEARS_BUILD_MODE", "YEARS_BUILD_MEDI",
    "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BEGINEXPLUATATION_MEDI",

    "OWN_CAR_AGE",

    # HOUSING
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",

    # AREA
    "TOTALAREA_MODE",

    # LIFTS
    "ELEVATORS_AVG", "ELEVATORS_MODE", "ELEVATORS_MEDI",

    # SIMPLE FLAGS
    "FLAG_MOBIL",
    "FLAG_CONT_MOBILE",

    # HIGH CORR
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "LIVINGAREA_MEDI",
    "APARTMENTS_MEDI",
    "AMT_GOODS_PRICE",
    "APARTMENTS_MODE",
    "LIVINGAREA_MODE",
    "REGION_RATING_CLIENT",
    "LIVINGAREA_AVG",
    "CNT_CHILDREN",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "LIVE_REGION_NOT_WORK_REGION",
    "LIVE_CITY_NOT_WORK_CITY",

    # USELESS (из ноутбука)
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "FLAG_EMAIL",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
]


def transform_application(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=DROP_COLS_STATIC, errors="ignore")

    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    doc_cols_to_drop = [c for c in doc_cols if c != "FLAG_DOCUMENT_3"]
    df = df.drop(columns=doc_cols_to_drop, errors="ignore")

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        denom = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / denom

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        denom = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / denom

    return df


def build_bureau_agg(bureau: pd.DataFrame) -> pd.DataFrame:
    bureau_agg = (
        bureau.groupby("SK_ID_CURR")
        .agg(
            BUREAU_CNT=("SK_ID_BUREAU", "count"),
            BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
            BUREAU_DAYS_CREDIT_AVG=("DAYS_CREDIT", "mean"),
            BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
            BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
            BUREAU_CNT_CREDIT_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
            BUREAU_AMT_CREDIT_SUM_DEBT_MAX=("AMT_CREDIT_SUM_DEBT", "max"),
            BUREAU_AMT_CREDIT_SUM_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
            BUREAU_AMT_CREDIT_SUM_SUM=("AMT_CREDIT_SUM", "sum"),
        )
        .reset_index()
    )

    status_cnt = pd.crosstab(bureau["SK_ID_CURR"], bureau["CREDIT_ACTIVE"]).reset_index()
    status_cnt = status_cnt.rename(
        columns={
            "Active": "BUREAU_ACTIVE_CNT",
            "Closed": "BUREAU_CLOSED_CNT",
            "Sold": "BUREAU_SOLD_CNT",
            "Bad debt": "BUREAU_BAD_DEBT_CNT",
        }
    )

    bureau_agg = bureau_agg.merge(status_cnt, on="SK_ID_CURR", how="left")
    return bureau_agg


def build_prev_app_agg(prev_app: pd.DataFrame) -> pd.DataFrame:
    prev = prev_app.copy()

    if {"AMT_CREDIT", "AMT_APPLICATION"}.issubset(prev.columns):
        prev["CREDIT_APP_RATIO"] = prev["AMT_CREDIT"] / prev["AMT_APPLICATION"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(prev.columns):
        prev["ANNUITY_CREDIT_RATIO"] = prev["AMT_ANNUITY"] / prev["AMT_CREDIT"].replace(0, np.nan)

    prev_app_agg = (
        prev.groupby("SK_ID_CURR")
        .agg(
            PREV_APP_CNT=("SK_ID_PREV", "count"),
            PREV_APP_AMT_CREDIT_SUM=("AMT_CREDIT", "sum"),
            PREV_APP_AMT_CREDIT_MAX=("AMT_CREDIT", "max"),
            PREV_APP_CREDIT_APP_RATIO_AVG=("CREDIT_APP_RATIO", "mean"),
            PREV_APP_ANNUITY_CREDIT_RATIO_AVG=("ANNUITY_CREDIT_RATIO", "mean"),
            PREV_APP_CNT_PAYMENT_AVG=("CNT_PAYMENT", "mean"),
            PREV_APP_LAST_DAYS_DECISION=("DAYS_DECISION", "max"),
            PREV_APP_ANNUITY_CREDIT_AVG=("AMT_ANNUITY", "mean"),
        )
        .reset_index()
    )

    st = pd.crosstab(prev["SK_ID_CURR"], prev["NAME_CONTRACT_STATUS"])
    st = st.rename(
        columns={
            "Approved": "PREV_APP_APPROVED_CNT",
            "Refused": "PREV_APP_REFUSED_CNT",
            "Canceled": "PREV_APP_CANCELED_CNT",
            "Unused offer": "PREV_APP_UNUSED_OFFER_CNT",
        }
    )

    for col in [
        "PREV_APP_APPROVED_CNT",
        "PREV_APP_REFUSED_CNT",
        "PREV_APP_CANCELED_CNT",
        "PREV_APP_UNUSED_OFFER_CNT",
    ]:
        if col not in st.columns:
            st[col] = 0

    st["PREV_APP_TOTAL_CNT"] = st[
        ["PREV_APP_APPROVED_CNT", "PREV_APP_REFUSED_CNT", "PREV_APP_CANCELED_CNT", "PREV_APP_UNUSED_OFFER_CNT"]
    ].sum(axis=1)

    denom = st["PREV_APP_TOTAL_CNT"].replace(0, np.nan)
    st["PREV_APP_APPROVED_RATE"] = st["PREV_APP_APPROVED_CNT"] / denom
    st["PREV_APP_REFUSED_RATE"] = st["PREV_APP_REFUSED_CNT"] / denom

    st = st.reset_index()[["SK_ID_CURR", "PREV_APP_APPROVED_RATE", "PREV_APP_REFUSED_RATE"]]

    prev_app_agg = prev_app_agg.merge(st, on="SK_ID_CURR", how="left")
    return prev_app_agg

