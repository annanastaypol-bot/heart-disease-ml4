"""Проверки D-01–D-08 и контроль идентификаторов между порциями.

Результат проверки: (принятый DataFrame, карантин DataFrame).
Функции не содержат медицинской логики и используются только в учебном проекте.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from features import CAT, CODES, NUM, RAW


class BatchRejected(ValueError):
    pass


def check_d01_schema(df: pd.DataFrame, labelled: bool = True) -> None:
    required = set(RAW + (['cardio'] if labelled else []))
    missing = sorted(required - set(df.columns))
    if missing:
        raise BatchRejected('D-01: отсутствуют столбцы: ' + ', '.join(missing))
    if df.empty:
        raise BatchRejected('D-01: пустая порция')


def check_d02_ids(df: pd.DataFrame, seen_ids=()) -> None:
    ids = pd.to_numeric(df['id'], errors='coerce')
    history = set(seen_ids)
    if ids.isna().any() or (~np.isfinite(ids)).any() or (ids % 1 != 0).any():
        raise BatchRejected('D-02: недопустимый id')
    if ids.duplicated().any() or ids.isin(history).any():
        raise BatchRejected('D-02: повтор id в порции или истории')


def check_d08_missing(df: pd.DataFrame) -> None:
    for col in CAT[1:]:
        if df[col].isna().mean() > 0.05:
            raise BatchRejected(f'D-08: доля пропусков {col} превышает 5%')


def quarantine_mask(df: pd.DataFrame, labelled: bool = True) -> pd.Series:
    """D-03 возраст, D-04 рост/масса, D-05 давление, D-06 коды, D-07 метка."""
    x = df.copy()
    cols = NUM + CAT + (['cardio'] if labelled else [])
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors='coerce')
    good = x[NUM + ['gender']].notna().all(axis=1)
    good &= x.age.div(365.25).between(18, 100)           # D-03
    good &= x.height.between(120, 220) & x.weight.between(30, 250) # D-04
    good &= x.ap_hi.between(70, 250) & x.ap_lo.between(40, 150) & (x.ap_hi > x.ap_lo) # D-05
    for col, codes in CODES.items():
        valid = x[col].isin(codes)
        if col != 'gender':
            valid |= x[col].isna()
        good &= valid                                      # D-06
    if labelled:
        good &= x.cardio.isin([0, 1])                    # D-07
    return ~good


def validate_batch(df: pd.DataFrame, seen_ids=(), labelled: bool = True):
    check_d01_schema(df, labelled)
    check_d02_ids(df, seen_ids)
    # Значения признаков должны быть числами (для необязательных значений пропуски разрешены).
    typed = df.copy()
    for col in RAW + (['cardio'] if labelled else []):
        orig = typed[col]
        typed[col] = pd.to_numeric(orig, errors='coerce')
        if (orig.notna() & typed[col].isna()).any():
            raise BatchRejected(f'D-01: нечисловое значение в {col}')
    check_d08_missing(typed)
    quarantine = quarantine_mask(typed, labelled=labelled)
    return typed.loc[~quarantine].copy(), typed.loc[quarantine].copy()


def load_batch(path: str, seen_ids=(), labelled=True):
    df = pd.read_excel(path) if str(path).lower().endswith('.xlsx') else pd.read_csv(path)
    return validate_batch(df, seen_ids=seen_ids, labelled=labelled)
