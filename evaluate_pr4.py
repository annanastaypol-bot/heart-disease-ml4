"""Практическая №4: воспроизводимая оценка сохранённой модели.

ВАЖНО: это заново подготовленный код ПО ОПИСАНИЮ отчёта, а не оригинальный
экспериментальный скрипт автора. Точные совпадения метрик зависят от тех же
обучающих/тестовых данных, порций и модели v1 из ПР №2/№3.

Пример: python evaluate_pr4.py --train data/train.csv --test data/test.csv \
    --model models/model_v1.joblib --batches data/batch_01.csv \
    data/batch_02.csv data/batch_03.csv data/batch_04.csv
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Не удалять: joblib-модель из ПР №3 может ссылаться на features.add_features.
import features  # noqa: F401
from features import NUM, CAT, make_pipeline
from quality_checks import load_batch, BatchRejected

SEED = 42
COLUMNS = NUM + CAT


def as_number(value):
    return None if value is None or not np.isfinite(value) else round(float(value), 6)


def scores(y, prob, threshold=0.5):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'n': int(len(y)), 'positives': int(y.sum()), 'threshold': float(threshold),
        'roc_auc': as_number(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else None,
        'ap': as_number(average_precision_score(y, prob)) if y.sum() else None,
        'precision': as_number(precision_score(y, pred, zero_division=0)),
        'recall': as_number(recall_score(y, pred, zero_division=0)),
        'f1': as_number(f1_score(y, pred, zero_division=0)),
        'accuracy': as_number(accuracy_score(y, pred)),
        'tp': int(tp), 'fn': int(fn), 'fp': int(fp), 'tn': int(tn),
        'mean_probability': as_number(prob.mean()),
        'positive_fraction': as_number(y.mean()),
        'predicted_high_fraction': as_number(pred.mean()),
    }


def choose_threshold(y, oof_prob):
    """Минимизировать 5*FN+FP; равные потери -> наибольший порог."""
    options=[]
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (oof_prob >= t).astype(int)
        fn = int(((np.asarray(y) == 1) & (pred == 0)).sum())
        fp = int(((np.asarray(y) == 0) & (pred == 1)).sum())
        options.append({'threshold':float(t), 'cost':int(5*fn+fp), 'fn':fn, 'fp':fp})
    best=min(options,key=lambda o:(o['cost'],-o['threshold']))
    return best, pd.DataFrame(options)


def bootstrap_auc(y, first, second, repeats=3000):
    """Стратифицированные парные 95% ДИ для ROC-AUC и разности."""
    y=np.asarray(y,dtype=int); first=np.asarray(first); second=np.asarray(second)
    pos=np.flatnonzero(y == 1); neg=np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0: return None
    rng=np.random.default_rng(SEED)
    val=np.empty((repeats,3),dtype=float)
    for i in range(repeats):
        selected=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        a=roc_auc_score(y[selected],first[selected]); b=roc_auc_score(y[selected],second[selected])
        val[i]=[a,b,a-b]
    out={}
    for j,name in enumerate(['forest','logreg','difference']):
        out[name]=[as_number(v) for v in np.quantile(val[:,j],[0.025,0.975])]
    # Двусторонний эмпирический bootstrap p, грубая эвристика.
    out['p_approx']=as_number(min(1.,2*min(np.mean(val[:,2]>=0),np.mean(val[:,2]<=0))))
    return out


def psi(reference, current):
    """PSI: три квантильных интервала по train, сглаживание +0,5 к частотам."""
    ref=pd.to_numeric(pd.Series(reference),errors='coerce').dropna().to_numpy()
    cur=pd.to_numeric(pd.Series(current),errors='coerce').dropna().to_numpy()
    if not len(ref) or not len(cur): return None
    cut=np.unique(np.quantile(ref,[0,1/3,2/3,1]))
    # Вырожденная шкала: нет трёх непустых границ.
    if len(cut)<4: return None
    bins=np.r_[-np.inf,cut[1:-1],np.inf]
    a=np.histogram(ref,bins=bins)[0].astype(float)+0.5
    b=np.histogram(cur,bins=bins)[0].astype(float)+0.5
    a/=a.sum(); b/=b.sum()
    return as_number(np.sum((b-a)*np.log(b/a)))


def plot_threshold(tab, threshold, path):
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(tab.threshold,tab.cost);ax.axvline(threshold,color='tab:orange',linestyle='--',label=f'Порог {threshold:.2f}')
    ax.set(xlabel='Порог классификации',ylabel='Стоимость 5×FN + 1×FP',title='Выбор порога по обучению')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def plot_confusion(y,prob,threshold,path):
    m=confusion_matrix(y,(np.asarray(prob)>=threshold).astype(int),labels=[0,1])
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(m,cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(m[i,j]),ha='center',va='center',fontsize=13)
    ax.set(xticks=[0,1],yticks=[0,1],xlabel='Прогноз',ylabel='Факт',title=f'Матрица ошибок, порог {threshold:.2f}')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def plot_curves(y,models,out):
    fig,ax=plt.subplots(figsize=(6,4))
    for name,prob in models.items():
        fpr,tpr,_=roc_curve(y,prob)
        ax.step(fpr,tpr,where='post',label=f'{name}: AUC {roc_auc_score(y,prob):.3f}')
    ax.plot([0,1],[0,1],'k--',alpha=.5);ax.legend()
    ax.set(xlabel='Доля FP',ylabel='Recall',title='ROC-кривые на тесте')
    fig.tight_layout();fig.savefig(out/'roc_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for name,prob in models.items():
        precision,recall,_=precision_recall_curve(y,prob)
        ax.step(recall,precision,where='post',label=f'{name}: AP {average_precision_score(y,prob):.3f}')
    ax.axhline(np.mean(y),color='gray',linestyle='--',label='Доля класса 1')
    ax.set(xlabel='Recall',ylabel='Precision',title='PR-кривые на тесте')
    ax.legend();fig.tight_layout();fig.savefig(out/'pr_curves.png',dpi=160);plt.close(fig)


def slice_scores(frame,prob,threshold):
    rows=[]
    median_age=float(frame.age.median())
    cuts=[('gender=1',frame.gender==1),('gender=2',frame.gender==2),
          ('age<=median',frame.age<=median_age),('age>median',frame.age>median_age)]
    for name,mask in cuts:
        if int(mask.sum()) == 0: continue
        rec=scores(frame.loc[mask,'cardio'].to_numpy(),np.asarray(prob)[mask.to_numpy()],threshold)
        rows.append({'slice':name,**rec})
    return rows


def robust_scores(model, frame, threshold):
    rng=np.random.default_rng(SEED)
    y=frame.cardio.astype(int).to_numpy()
    X=frame[COLUMNS]
    checks={'original':X.copy()}
    noisy=X.copy()
    for col in ['height','weight','ap_hi','ap_lo']:
        noisy[col]=noisy[col]*(1+rng.normal(0,.02,size=len(noisy)))
    checks['noise_2pct']=noisy
    missing=X.copy()
    k=max(1,round(.10*len(missing)))
    missing.iloc[rng.choice(len(missing),k,replace=False),missing.columns.get_loc('cholesterol')]=np.nan
    checks['cholesterol_10pct_missing']=missing
    return {name:scores(y,model.predict_proba(x)[:,1],threshold) for name,x in checks.items()}


def permutation_scores(model,test,output):
    X=test[COLUMNS];y=test.cardio.astype(int)
    p=permutation_importance(model,X,y,scoring='roc_auc',n_repeats=40,random_state=SEED,n_jobs=1)
    order=np.argsort(p.importances_mean)
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.barh(np.asarray(COLUMNS)[order],p.importances_mean[order],xerr=p.importances_std[order])
    ax.set(xlabel='Уменьшение ROC-AUC',title='Перестановочная важность (тест)')
    fig.tight_layout();fig.savefig(output/'permutation_importance.png',dpi=160);plt.close(fig)
    return [{'feature':COLUMNS[i],'mean_auc_drop':as_number(p.importances_mean[i]),
             'std':as_number(p.importances_std[i])} for i in np.argsort(p.importances_mean)[::-1]]


def partial_dependence_manual(model, train, output):
    """Учебная частичная зависимость: подмена признака во всех строках train."""
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    result={}
    for ax,col in zip(axes,['ap_hi','age','weight']):
        vals=np.unique(np.quantile(train[col].dropna(),[.05,.25,.5,.75,.95]))
        estimates=[]
        for val in vals:
            tmp=train[COLUMNS].copy();tmp[col]=val
            estimates.append(float(model.predict_proba(tmp)[:,1].mean()))
        ax.plot(vals,estimates,marker='o');ax.set(xlabel=col,ylabel='Средняя p',title=col)
        result[col]=[{'value':as_number(a),'mean_probability':as_number(b)} for a,b in zip(vals,estimates)]
    fig.tight_layout();fig.savefig(output/'partial_dependence.png',dpi=160);plt.close(fig)
    return result


def local_explanations(model,train,test,threshold):
    """По одному верному и ошибочному примеру; замена на медиану/моду train."""
    p=model.predict_proba(test[COLUMNS])[:,1]
    pred=(p>=threshold).astype(int)
    y=test.cardio.astype(int).to_numpy()
    choices=[]
    for title,mask in [('correct',pred==y),('incorrect',pred!=y)]:
        ix=np.flatnonzero(mask)
        if not len(ix): continue
        i=int(ix[0]);base=float(p[i]);rec={}
        for col in COLUMNS:
            copy=test[COLUMNS].iloc[[i]].copy()
            typical=train[col].median() if col in NUM else train[col].mode().iloc[0]
            copy[col]=typical
            rec[col]=as_number(base-float(model.predict_proba(copy)[:,1][0]))
        choices.append({'kind':title,'row_index':i,'actual':int(y[i]),'predicted':int(pred[i]),
                        'probability':as_number(base),'feature_changes':rec})
    return choices


def batch_analysis(model,train,batchpaths,threshold,out):
    seen=set(train.id)
    output=[]
    for idx,path in enumerate(batchpaths,1):
        accepted,quarantine=load_batch(str(path),seen_ids=seen,labelled=True)
        if len(quarantine):
            print(f'Внимание: {path}: в карантине {len(quarantine)} строк')
        if accepted.empty: continue
        prob=model.predict_proba(accepted[COLUMNS])[:,1]
        y=accepted.cardio.astype(int)
        r=scores(y,prob,threshold)
        r.update({'batch':Path(path).name,'quarantine':int(len(quarantine)),
                  'psi_ap_hi':psi(train.ap_hi,accepted.ap_hi),
                  'psi_ap_lo':psi(train.ap_lo,accepted.ap_lo),
                  'psi_age':psi(train.age,accepted.age)})
        output.append(r)
        seen.update(accepted.id)
    if output:
        pd.DataFrame(output).to_csv(out/'batch_metrics.csv',index=False)
        df=pd.DataFrame(output)
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        axes[0].plot(df.batch,df.roc_auc,marker='o',label='ROC-AUC')
        axes[0].axhline(.7,linestyle='--',alpha=.4,label='Граница 0.70')
        axes[0].set(title='Качество порций',ylabel='ROC-AUC');axes[0].legend()
        for c in ['psi_ap_hi','psi_ap_lo','psi_age']:
            axes[1].plot(df.batch,df[c],marker='o',label=c)
        axes[1].axhline(.25,linestyle='--',alpha=.4)
        axes[1].set(title='Изменение данных',ylabel='PSI');axes[1].legend()
        for ax in axes:ax.tick_params(axis='x',rotation=20)
        fig.tight_layout();fig.savefig(out/'batch_monitoring.png',dpi=160);plt.close(fig)
    return output


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    train,badtrain=load_batch(args.train,labelled=True)
    test,badtest=load_batch(args.test,seen_ids=set(train.id),labelled=True)
    if len(badtrain) or len(badtest):
        raise BatchRejected('Найдены строки карантина train/test; проверьте исходное разделение')
    model=joblib.load(args.model)  # load ONLY a trusted model, never an untrusted pickle.
    X=train[COLUMNS];y=train.cardio.astype(int)
    Xt=test[COLUMNS];yt=test.cardio.astype(int)
    if len(np.unique(yt))<2: raise ValueError('В тесте нужны оба класса для ROC-AUC')
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    oof=cross_val_predict(clone(model),X,y,cv=cv,method='predict_proba')[:,1]
    threshold,grid=choose_threshold(y,oof)
    grid.to_csv(out/'threshold_grid.csv',index=False)
    plot_threshold(grid,threshold['threshold'],out/'threshold_selection.png')
    forest_prob=model.predict_proba(Xt)[:,1]
    logreg=make_pipeline(LogisticRegression(C=.01,penalty='l2',solver='liblinear',max_iter=3000,random_state=SEED)).fit(X,y)
    logreg_prob=logreg.predict_proba(Xt)[:,1]
    dummy=DummyClassifier(strategy='most_frequent').fit(X,y)
    dummy_prob=dummy.predict_proba(Xt)[:,1]
    main={
        'training_rows':int(len(train)),'test_rows':int(len(test)),
        'threshold_oof_train':threshold,
        'dummy':scores(yt,dummy_prob,0.5),
        'logistic_regression':scores(yt,logreg_prob,0.5),
        'forest_050':scores(yt,forest_prob,0.5),
        'forest_selected':scores(yt,forest_prob,threshold['threshold']),
        'auc_bootstrap_95pct':bootstrap_auc(yt,forest_prob,logreg_prob),
        'slices':slice_scores(test,forest_prob,threshold['threshold']),
        'robustness':robust_scores(model,test,threshold['threshold']),
        'permutation_importance':permutation_scores(model,test,out),
        'partial_dependence':partial_dependence_manual(model,train,out),
        'local_explanations':local_explanations(model,train,test,threshold['threshold']),
    }
    plot_confusion(yt,forest_prob,threshold['threshold'],out/'confusion_matrix.png')
    plot_curves(yt,{'Random Forest':forest_prob,'LogReg':logreg_prob},out)
    if args.batches:
        main['batches']=batch_analysis(model,train,args.batches,threshold['threshold'],out)
    with (out/'metrics.json').open('w',encoding='utf-8') as f:
        json.dump(main,f,ensure_ascii=False,indent=2)
    print('Результаты записаны в',out.resolve())
    print('Тестовый ROC-AUC леса:',main['forest_selected']['roc_auc'])
    print('Порог по обучающей выборке:',threshold['threshold'])


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',default='data/train.csv')
    p.add_argument('--test',default='data/test.csv')
    p.add_argument('--model',default='models/model_v1.joblib')
    p.add_argument('--batches',nargs='*',default=[])
    p.add_argument('--out',default='results')
    return p.parse_args()


if __name__=='__main__':
    run(parse())
