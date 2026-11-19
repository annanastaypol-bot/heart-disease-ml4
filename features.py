"""Подготовка признаков из проекта практической работы №3.

Этот файл необходим в том числе для joblib.load(model_v1.joblib), поскольку
сериализованный sklearn Pipeline ссылается на функцию features.add_features.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUM = ['age','height','weight','ap_hi','ap_lo']
CAT = ['gender','cholesterol','gluc','smoke','alco','active']
EXTRA = ['age_years','bmi','pulse_pressure','mean_ap']
RAW = ['id'] + NUM + CAT
CODES = {'gender':[1,2], 'cholesterol':[1,2,3], 'gluc':[1,2,3], 'smoke':[0,1], 'alco':[0,1], 'active':[0,1]}


def load_checked(path, training=False):
    """D-01–D-08: отклонение неверной схемы/ID/пропусков; карантин плохих строк."""
    path = str(path)
    df = pd.read_excel(path) if path.lower().endswith('.xlsx') else pd.read_csv(path)
    required = RAW + (['cardio'] if training else [])
    if not set(required).issubset(df.columns):
        raise ValueError('Missing columns: '+str(set(required)-set(df.columns)))
    if df.empty: raise ValueError('Empty file')
    df = df.copy()
    for c in required:
        original = df[c].copy()
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if (original.notna() & df[c].isna()).any():
            raise ValueError('Invalid numeric encoding in '+c)
    ids = df.id
    if ids.isna().any() or not np.isfinite(ids).all() or (ids%1 != 0).any() or ids.duplicated().any():
        raise ValueError('Invalid or duplicate id (D-02)')
    for c in CAT[1:]:
        if df[c].isna().mean() > .05: raise ValueError('D-08: missing >5%: '+c)
    ok = df[NUM+['gender']].notna().all(axis=1)
    ok &= df.age.div(365.25).between(18,100) & df.height.between(120,220) & df.weight.between(30,250)
    ok &= df.ap_hi.between(70,250) & df.ap_lo.between(40,150) & (df.ap_hi>df.ap_lo)
    for c, allowed in CODES.items():
        ok &= df[c].isin(allowed) | (df[c].isna() if c!='gender' else False)
    if training: ok &= df.cardio.isin([0,1])
    return df.loc[ok].copy(), df.loc[~ok].copy()


def add_features(X):
    X = X.copy()
    X['age_years'] = X.age/365.25
    X['bmi'] = X.weight/(X.height/100)**2
    X['pulse_pressure'] = X.ap_hi-X.ap_lo
    X['mean_ap'] = X.ap_lo+X.pulse_pressure/3
    return X


def make_pipeline(model, extended=True, scale=True):
    numeric = [('impute',SimpleImputer(strategy='median'))]
    if scale: numeric.append(('scale', StandardScaler()))
    cat = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(categories=[CODES[c] for c in CAT],handle_unknown='error',sparse_output=False))])
    prep = ColumnTransformer([('num',Pipeline(numeric),NUM+(EXTRA if extended else [])),('cat',cat,CAT)])
    return Pipeline([('features',FunctionTransformer(add_features if extended else None)),('prep',prep),('model',model)])
