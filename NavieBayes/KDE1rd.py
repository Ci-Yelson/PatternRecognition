import numpy as np
import matplotlib.pyplot as plt
from ROC import *

class KDE_NavieBayes:
    def __init__(self,x_tr1,x_tr2,x_te1,x_te2):
        self.x_tr1 = x_tr1
        self.x_tr2 = x_tr2
        self.x_te1 = x_te1
        self.x_te2 = x_te2

        self.bw1 = None     #窗口大小 
        self.bw2 = None     #窗口大小

        self.rpw1 = x_tr1.shape[0] / (x_tr1.shape[0]+x_tr2.shape[0])
        self.rpw2 = 1 - self.rpw1

        self.D = self.x_tr1.shape[1]  #特征维数

    def norm_kernel(self,x):
        return 1/np.sqrt(2*np.pi) * np.exp(-0.5*np.square(x))
        
    def gxi(self,x,data,bw,pwi):
        '''
        x:(D,N)
        data:(D,N)
        '''
        x = x.reshape(self.D,-1)
        data = data.reshape(-1,self.D).T
        y_norm = 1
        for i in range(self.D):
            y_norm = y_norm * self.norm_kernel((x[i]-data[i])/bw)
        y = np.sum(y_norm) / (data.shape[1]*np.power(bw,self.D))
        return float(y*pwi)

    def gx(self,x,data1,data2,bw,pw1,pw2):
        gx = self.gxi(x,data1,bw,pw1) - self.gxi(x,data2,bw,pw2)
        return float(gx)
    
    def kde(self,x,bw=[2],pwi=1):
        x = np.sort(x)
        y = []

        for i in x:
            _y = self.gxi(i,x,bw,pwi)
            y.append(_y)

        plt.plot(x,y,linewidth=2)


    def train(self,show_arg=False):

        # bindwidth - 经验法则
        # self.bw1,self.bw2 = [],[]
        x1,x2 = self.x_tr1.T,self.x_tr2.T
        for i in range(self.D):
            sig1 = np.sum(np.square(x1[i]-np.mean(x1[i]))) / x1[i].shape[0]
            sig1 = np.sqrt(sig1)
            scott1 = 1.06 * sig1 * x1[i].shape[0] ** (-1./(1+4))
            sig2 = np.sum(np.square(x2[i]-np.mean(x2[i]))) / x2[i].shape[0]
            sig2 = np.sqrt(sig2)
            scott2 = 1.06 * sig2 * x2[i].shape[0] ** (-1./(1+4))
            # print(scott1)
            # print(scott2)
            self.bw1 = scott1
            self.bw2 = scott2

        if show_arg == True:
            x1,x2 = self.x_tr1.T[0],self.x_tr2.T[0]

            plt.figure(figsize=(16,8))

            plt.subplot(1,2,1)
            plt.hist(x1,bins=int(np.max(x1)-np.min(x1)),density=True,alpha=0.5,label='身高-男生')
            plt.hist(x2,bins=int(np.max(x2)-np.min(x2)),density=True,alpha=0.5,label='身高-女生')
            # draw
            self.kde(x1,bw=scott1)
            self.kde(x2,bw=scott2)
            plt.title('类条件密度估计(by implementation)',fontsize=18)

            # sns
            plt.subplot(1,2,2)
            plt.hist(x1,bins=int(np.max(x1)-np.min(x1)),density=True,alpha=0.5,label='身高-男生')
            plt.hist(x2,bins=int(np.max(x2)-np.min(x2)),density=True,alpha=0.5,label='身高-女生')

            import seaborn as sns
            sns.kdeplot(x1, shade=True, linewidth=2, bw_method='scott')
            sns.kdeplot(x2, shade=True, linewidth=2, bw_method='scott')
            plt.title('类条件密度估计(by seaborn)',fontsize=18)

            plt.show()


    def classify(self,pw1,pw2):
        x1,x2 = self.x_tr1.reshape(1,-1)[0],self.x_tr2.reshape(1,-1)[0]

        y_pe1 = np.array([self.gx(i,data1=x1,data2=x2,bw=self.bw1,pw1=pw1,pw2=pw2)>0 for i in self.x_te1])
        y_pe2 = np.array([self.gx(i,data1=x1,data2=x2,bw=self.bw1,pw1=pw1,pw2=pw2)>0 for i in self.x_te2])
        acc1 = np.mean(y_pe1)
        acc2 = 1-np.mean(y_pe2)
        print('pw1:{0:.2f}, pw2:{1:.2f}'.format(pw1,pw2))
        print('男生正确率：({0} / {1}) = '.format(np.sum(y_pe1),y_pe1.shape[0]),acc1)
        print('女生正确率：({0} / {1}) = '.format(y_pe2.shape[0]-np.sum(y_pe2),y_pe2.shape[0]),acc2)

        self.kde(x1,pwi=pw1,bw=self.bw1)
        self.kde(x2,pwi=pw2,bw=self.bw2)
        plt.title('pw1={0:.2f}, pw2={1:.2f}'.format(pw1,pw2))

        x0 = 0
        for i in np.arange(150,180,0.01):
            gx1 = self.gxi(i,data=x1,bw=self.bw1,pwi=pw1)
            gx2 = self.gxi(i,data=x2,bw=self.bw2,pwi=pw2)
            if abs(gx1-gx2)<=1e-4:
                x0 = i
                break
        
        plt.vlines(x0,0,0.15,linewidth=2,label='分界点：{0:.3f}'.format(x0),colors='r')
        plt.legend()

    def ROC_curve(self,pw1,pw2):
        x1,x2 = self.x_tr1.reshape(1,-1)[0],self.x_tr2.reshape(1,-1)[0]
        score1 = np.array([self.gx(i,data1=x1,data2=x2,bw=self.bw1,pw1=pw1,pw2=pw2) for i in self.x_te1])
        score2 = np.array([self.gx(i,data1=x1,data2=x2,bw=self.bw1,pw1=pw1,pw2=pw2) for i in self.x_te2])
        score = np.concatenate([score1,score2])
        y = np.concatenate([np.ones(self.x_te1.shape[0]),np.zeros(self.x_te2.shape[0])])
        _fpr,_tpr,_auc = ROC_method1(y,score)

        return _fpr,_tpr,_auc

    def ERR_curve(self,pw1,pw2):
        x1,x2 = self.x_tr1.reshape(1,-1)[0],self.x_tr2.reshape(1,-1)[0]
        score1 = np.array([self.gx(i,data1=x1,data2=x2,bw=self.bw1,pw1=pw1,pw2=pw2) for i in self.x_te1])
        score2 = np.array([self.gx(i,data1=x1,data2=x2,bw=self.bw1,pw1=pw1,pw2=pw2) for i in self.x_te2])
        score = np.concatenate([score1,score2])
        y = np.concatenate([np.ones(self.x_te1.shape[0]),np.zeros(self.x_te2.shape[0])])
        _fpr,_tpr = ERR(y,score)

        return _fpr,_tpr