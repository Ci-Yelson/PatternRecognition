import numpy as np
import pandas as pd
from pandas import DataFrame

def ROC_method1(y,score):
    '''
    y:(Num,) 取值[0,1]
    socre:(Num,)
    '''
    pos = np.vstack([np.ones((y[y==1]).shape),score[y==1]]).T
    neg = np.vstack([np.zeros((y[y==0]).shape),score[y==0]]).T
    # print("Pos data shape:",pos.shape)
    # print("Neg data shape:",neg.shape)
    pn = np.vstack([pos,neg])
    pn = DataFrame(data=pn)

    pn.sort_values(by=1,inplace=True,ascending=False)

    # Draw ROC curve and calc AUC
    fp,tp = 0,0
    pre_score = np.min(pn[1]) - 1
    fpr,tpr = [],[]

    auc = 0
    pre_fp,pre_tp = 0,0

    def calcarea(x1,x2,y1,y2):
        return abs(x2-x1)*(y1+y2)/2

    for index, row in pn.iterrows():
        score = row[1]
        # if abs(pre_score - score) > 1e-3:
        if pre_score != score:
            auc += calcarea(fp,pre_fp,tp,pre_tp)
            pre_fp,pre_tp = fp,tp
            fpr.append(fp/neg.shape[0]) 
            tpr.append(tp/pos.shape[0]) 
            pre_score = score

        if row[0] == 1:
            tp+=1
        else:
            fp+=1

    fpr.append(fp/neg.shape[0]) 
    tpr.append(tp/pos.shape[0]) 
    auc += calcarea(neg.shape[0],pre_fp,neg.shape[0],pre_tp)
    auc /= pos.shape[0]*neg.shape[0]
    # print("Total data num:",pn.shape[0])
    # print("AUC:",auc)

    return fpr,tpr,auc

def ROC_method2(y,score):
    '''
    y:(Num,) 取值[0,1]
    socre:(Num,)
    '''
    mi,mx = np.min(score), np.max(score)
    d = (mx-mi)/100

    fpr,tpr = [0],[0]

    P = np.sum(y)
    N = y.shape[0]-P
    curScore = mx
    for i in range(100):
        _tpr = np.sum(y[score >= curScore]) / P
        _fpr = np.sum(1-y[score >= curScore]) / N
        fpr.append(_fpr)
        tpr.append(_tpr)
        curScore -= d
    
    fpr.append(1)
    tpr.append(1)

    auc = 0
    for i in range(1,len(fpr)):
        auc += (tpr[i]+tpr[i-1])/2 * (fpr[i]-fpr[i-1])
    
    return fpr,tpr,auc

def ERR(y,score):
    '''
    y:(Num,) 取值[0,1]
    socre:(Num,)
    '''
    # pos = np.vstack([np.ones((y[y==1]).shape),score[y==1]]).T
    # neg = np.vstack([np.zeros((y[y==0]).shape),score[y==0]]).T
    mi,mx = np.min(score), np.max(score)
    d = (mx-mi)/100

    far,frr = [],[]

    P = np.sum(y)
    N = y.shape[0]-P
    curScore = mx
    for i in range(100):
        fp = np.sum(1-y[score >= curScore])
        tn = np.sum(1-y[score < curScore])
        fn = np.sum(y[score < curScore])
        tp = np.sum(y[score >= curScore])
        _far = fp / (fp+tn)
        _frr = fn / (fn+tp)
        far.append(_far)
        frr.append(_frr)
        curScore -= d
    
    
    return far,frr