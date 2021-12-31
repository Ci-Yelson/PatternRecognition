import numpy as np
import matplotlib.pyplot as plt
from ROC import *

# 一维最大似然估计
class MLE1rd_NavieBayes:
    def __init__(self,x_tr1,x_tr2,x_te1,x_te2):
        self.x_tr1 = x_tr1
        self.x_tr2 = x_tr2
        self.x_te1 = x_te1
        self.x_te2 = x_te2

        self.u = None       #两类的均值
        self.sig = None     #两类的方差矩阵

        self.rpw1 = x_tr1.shape[0] / (x_tr1.shape[0]+x_tr2.shape[0])
        self.rpw2 = 1 - self.rpw1

        self.D = self.x_tr1.shape[1]  #特征维数


    def train(self,show_arg=False):
        
        C = 2
        self.u = [np.zeros(self.D) for i in range(C)]
        self.sig = [np.zeros((self.D,self.D)) for i in range(C)]

        self.u[0] = np.mean(self.x_tr1,axis=0).reshape(-1,1)
        self.u[1] = np.mean(self.x_tr2,axis=0).reshape(-1,1)

        # self.sig[0] = (self.x_tr1.T-self.u[0]).dot((self.x_tr1.T-self.u[0]).T) / self.x_tr1.shape[0]
        # self.sig[1] = (self.x_tr2.T-self.u[1]).dot((self.x_tr2.T-self.u[1]).T) / self.x_tr2.shape[0]
        self.sig[0] = np.cov(self.x_tr1.T)
        self.sig[1] = np.cov(self.x_tr2.T)

        if show_arg == True:
            print('============MLE============')
            print('u1:',self.u[0])
            print('sig1:',self.sig[0])
            print('u2:',self.u[1])
            print('sig2:',self.sig[1])
            print('============MLE============')


    # 正态分布形式
    def gxi(self,x,u,sig,pwi):
        # x = x.reshape(-1,1) # (D,)=>(D,1)
        p1 = 1/(np.sqrt(2*np.pi)*np.sqrt(sig))
        p2 = np.exp(-0.5*np.square((x-u)/np.sqrt(sig)))
        return float(p1 * p2 * pwi)

    def gx(self,x,pw1,pw2):
        gx = self.gxi(x,self.u[0],self.sig[0],pw1) - self.gxi(x,self.u[1],self.sig[1],pw2)
        return float(gx)

    def get_tick(self,pw1,pw2):
        x1,x2 = np.sort(self.x_te1,axis=0),np.sort(self.x_te2,axis=0)

        y1 = [self.gxi(i,self.u[0],self.sig[0],pw1) for i in x1]
        y2 = [self.gxi(i,self.u[1],self.sig[1],pw2) for i in x2]
        
        x0 = 0
        L,R = 160,170
        while 1:
            mid = (L+R)/2
            gx0 = self.gx([mid],pw1,pw2)
            if abs(gx0)<=1e-6:
                x0=mid
                break
            elif gx0>0:
                R=mid
            elif gx0<0:
                L=mid

        return x1,x2,y1,y2,x0

    def classify(self,pw1=None,pw2=None):
        
        if pw1 == None:
            pw1 = self.rpw1
        if pw2 == None:
            pw2 = self.rpw2

        y_pe1 = np.array([self.gx(i,pw1,pw2)>0 for i in self.x_te1])
        y_pe2 = np.array([self.gx(i,pw1,pw2)>0 for i in self.x_te2])
        acc1 = np.mean(y_pe1)
        acc2 = 1-np.mean(y_pe2)
        print('pw1:{0:.2f}, pw2:{1:.2f}'.format(pw1,pw2))
        print('男生正确率：({0} / {1}) = '.format(np.sum(y_pe1),y_pe1.shape[0]),acc1)
        print('女生正确率：({0} / {1}) = '.format(y_pe2.shape[0]-np.sum(y_pe2),y_pe2.shape[0]),acc2)

        # 绘制决策面
        x1,x2,y1,y2,x0 = self.get_tick(pw1,pw2)

        plt.plot(x1,y1,linewidth=2,label='男生：p(x|w1)p(w1)')
        plt.plot(x2,y2,linewidth=2,label='女生：p(x|w2)p(w2)')
        plt.vlines(x0,0,np.max(np.concatenate([y1,y2])),linewidth=2,label='分界点：{0:.3f}'.format(x0),colors='r')
        plt.title('pw1={0:.2f}, pw2={1:.2f}'.format(pw1,pw2))
        plt.legend()
        

    def ROC_curve(self,pw1,pw2):
        score1 = np.array([self.gx(i,pw1,pw2) for i in self.x_te1])
        score2 = np.array([self.gx(i,pw1,pw2) for i in self.x_te2])
        score = np.concatenate([score1,score2])
        y = np.concatenate([np.ones(self.x_te1.shape[0]),np.zeros(self.x_te2.shape[0])])
        _fpr,_tpr,_auc = ROC_method2(y,score)

        return _fpr,_tpr,_auc

    def ERR_curve(self,pw1,pw2):
        score1 = np.array([self.gx(i,pw1,pw2) for i in self.x_te1])
        score2 = np.array([self.gx(i,pw1,pw2) for i in self.x_te2])
        score = np.concatenate([score1,score2])
        y = np.concatenate([np.ones(self.x_te1.shape[0]),np.zeros(self.x_te2.shape[0])])
        _fpr,_tpr = ERR(y,score)

        return _fpr,_tpr


        