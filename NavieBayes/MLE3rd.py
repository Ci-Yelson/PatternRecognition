import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as process
from mpl_toolkits.mplot3d import Axes3D

from sklearn import metrics
from ROC import *


class MLE3rd_NavieBayes:
    def __init__(self,x_tr1,x_tr2,x_te1,x_te2):
    
        self.u = None       #两类的均值
        self.sig = None     #两类的方差矩阵

        self.rpw1 = x_tr1.shape[0] / (x_tr1.shape[0]+x_tr2.shape[0])
        self.rpw2 = 1 - self.rpw1

        self.D = x_tr1.shape[1]  #特征维数

        # # 归一化
        # scaler = process.MinMaxScaler(feature_range=[-1,1])
        # x_tr = np.concatenate([x_tr1,x_tr2])
        # x_te = np.concatenate([x_te1,x_te2])
        # x = np.concatenate([x_tr,x_te]).T

        # for i in range(self.D):
        #     x[i] = scaler.fit_transform(x[i].reshape(-1,1)).T[0]

        # self.x_tr1 = x[:,:x_tr1.shape[0]].T
        # self.x_tr2 = x[:,x_tr1.shape[0]:x_tr.shape[0]].T
        # self.x_te1 = x[:,x_tr.shape[0]:x_te1.shape[0]].T
        # self.x_te2 = x[:,x_te1.shape[0]:].T

        # print(x_tr1.shape)
        # print(x_tr2.shape)
        # print(x_te1.shape)
        # print(x_te2.shape)

        self.x_tr1 = x_tr1
        self.x_tr2 = x_tr2
        self.x_te1 = x_te1
        self.x_te2 = x_te2


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


    # 对数形式
    # def gxi(self,x,u,sig,pwi):
    #     x = x.reshape(-1,1)
    #     p1 = 0.5*self.D*np.log(2*np.pi)
    #     p2 = 0.5*np.log(np.linalg.det(sig))
    #     p3 = 0.5*(x-u).T@np.linalg.inv(sig)@(x-u)
    #     p4 = np.log(pwi)
    #     return p1-p2-p3+p4

    # 正态分布形式
    def gxi(self,x,u,sig,pwi):
        x = x.reshape(-1,1) # (D,)=>(D,1)
        p1 = 1/(np.power(2*np.pi,self.D/2)*np.sqrt(np.linalg.det(sig)))
        p2 = np.exp(-0.5 * (x-u).T @ np.linalg.inv(sig) @ (x-u))
        return p1 * p2 * pwi

    def gx(self,x,pw1,pw2):
        gx = self.gxi(x,self.u[0],self.sig[0],pw1) - self.gxi(x,self.u[1],self.sig[1],pw2)
        return float(gx)


    def classify(self,pw1,pw2,cnt):

        def paint(pw1,pw2):
            
            u1 = self.u[0]
            u2 = self.u[1]
            sig1 = self.sig[0]
            sig2 = self.sig[1]

            W1 = -0.5 * np.linalg.inv(sig1)
            W2 = -0.5 * np.linalg.inv(sig2)
            w1 = W1 - W2

            W1_1 = np.linalg.inv(sig1) @ u1
            W2_1 = np.linalg.inv(sig2) @ u2
            w2 = W1_1 - W2_1

            W1_0 = -0.5*(u1.T @ np.linalg.inv(sig1) @ u1) - 0.5*np.log(np.linalg.det(sig1)) + np.log(pw1)
            W2_0 = -0.5*(u2.T @ np.linalg.inv(sig2) @ u2) - 0.5*np.log(np.linalg.det(sig2)) + np.log(pw2)
            w3 = W1_0 - W2_0


            x1 = np.arange(150,190,0.01)
            x2 = np.arange(40,80,0.01)
            x1,x2 = np.meshgrid(x1,x2)

            a = w1[2,2]
            b = ((w1[0,2]+w1[2,0])*x1 + (w1[1,2]+w1[2,1])*x2 + w2[2])
            c = (w1[0,0]*x1*x1 + w1[1,1]*x2*x2 + (w1[0,1]+w1[1,0])*x1*x2 + w2[0]*x1 + w2[1]*x2) + w3


            y1 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
            y2 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

            # print(y1)
            ax = plt.subplot(121,projection='3d')
            ax.scatter(self.x_te1[:,0],self.x_te1[:,1],self.x_te1[:,2],color='r')
            ax.scatter(self.x_te2[:,0],self.x_te2[:,1],self.x_te2[:,2],color='b')
            # ax.plot_surface(x1,x2,y1,color='y',alpha=0.75)
            # ax.plot_surface(x1,x2,y2,color='y',alpha=0.75)
            ax.plot_surface(x1,x2,y1,cmap='rainbow',alpha=0.75)
            ax.plot_surface(x1,x2,y2,cmap='rainbow',alpha=0.75)
            # plt.tight_layout()

        y_pe1 = np.array([self.gx(i,pw1,pw2)>0 for i in self.x_te1])
        y_pe2 = np.array([self.gx(i,pw1,pw2)>0 for i in self.x_te2])
        acc1 = np.mean(y_pe1)
        acc2 = 1-np.mean(y_pe2)
        print('pw1:{0:.2f}, pw2:{1:.2f}'.format(pw1,pw2))
        print('男生正确率：({0} / {1}) = '.format(np.sum(y_pe1),y_pe1.shape[0]),acc1)
        print('女生正确率：({0} / {1}) = '.format(y_pe2.shape[0]-np.sum(y_pe2),y_pe2.shape[0]),acc2)

        plt.figure(figsize=(12,6))
        paint(pw1,pw2)

        x = np.concatenate([self.x_te1,self.x_te2])
        y = np.concatenate([np.ones(self.x_te1.shape[0]),np.zeros(self.x_te2.shape[0])])
        pred = np.array([self.gx(i,pw1,pw2) for i in x])
        fpr,tpr,auc = ROC_method2(y,pred)

        ax = plt.subplot(122)
        ax.plot(fpr,tpr)
        ax.plot([0,1], [0,1],linewidth=2,linestyle='dashed',c='grey')
        # ax.set_xlim([0,1])
        # ax.set_ylim([0,1])
        ax.set_xlabel('False positive rate',fontsize=18)
        ax.set_ylabel('True positive rate',fontsize=18)
        plt.suptitle('pw1={0:.2f}, pw2={1:.2f} 下的决策面'.format(pw1,pw2),fontsize=18)
        plt.show()



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
