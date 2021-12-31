import numpy as np
import matplotlib.pyplot as plt
from ROC import *

# 二维最大似然估计
class MLE2rd_NavieBayes:
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

    def classify(self,pw1,pw2):

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

            xx = np.arange(150,190,0.01)

            a = w1[1,1]
            b = (w1[1,0]+w1[0,1])*xx+w2[1]
            c = w1[0,0]*xx*xx + w2[0]*xx + w3

            y1 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
            y2 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

            y1 = y1[0]
            y2 = y2[0]

            # x_te = np.concatenate([self.x_te1,self.x_te2])
            # x1,x2 = np.meshgrid(np.sort(x_te[:,0]),np.sort(x_te[:,1]))
            N = 200
            x1 = np.linspace(140,190,N)
            x2 = np.linspace(30,100,N)
            x1,x2 = np.meshgrid(x1,x2)
            
            z = np.zeros(x1.shape)
            for i in range(x1.shape[0]):
                for j in range(x1.shape[1]):
                    px = np.array([x1[i,j],x2[i,j]]).reshape(-1,1)
                    z[i][j] = self.gx(px,pw1,pw2)

            plt.figure(figsize=(18,6))
            # print(pr,pc,cnt*3)
            ax1 = plt.subplot(131)
            # 测试集数据分布 + 决策面（曲线）
            ax1.plot(xx,y1,c='r')
            ax1.plot(xx,y2,c='r')
            ax1.scatter(self.x_te1[:,0],self.x_te1[:,1],color='r')
            ax1.scatter(self.x_te2[:,0],self.x_te2[:,1],color='b')

            ax2 = plt.subplot(132)
            # 取样数据等高线
            ax2.contourf(x1,x2,z,8,alpha=0.75,cmap=plt.cm.hot)
            # ax2.contour(x1,x2,z,8,zdir='z',offset=-0.0001,cmap='rainbow')
            # zdir : 'z' | 'x' | 'y' 表示把等高线图投射到哪个面
            # offset : 表示等高线图投射到指定页面的某个刻度
            C = ax2.contour(x1,x2,z,8,alpha=0.75,colors='black')
            plt.clabel(C,inline=True,fontsize=16)

            ax3 = plt.subplot(133,projection='3d')
            # 取样数据判别函数三维图
            ax3.plot_surface(x1,x2,z,rstride=2, cstride=1, cmap=plt.cm.Spectral)
            plt.suptitle('pw1={0:.2f}, pw2={1:.2f} 下的决策面(椭圆线)'.format(pw1,pw2),fontsize=18)
            plt.show()

        y_pe1 = np.array([self.gx(i,pw1,pw2)>0 for i in self.x_te1])
        y_pe2 = np.array([self.gx(i,pw1,pw2)>0 for i in self.x_te2])
        acc1 = np.mean(y_pe1)
        acc2 = 1-np.mean(y_pe2)
        print('pw1:{0:.2f}, pw2:{1:.2f}'.format(pw1,pw2))
        print('男生正确率：({0} / {1}) = '.format(np.sum(y_pe1),y_pe1.shape[0]),acc1)
        print('女生正确率：({0} / {1}) = '.format(y_pe2.shape[0]-np.sum(y_pe2),y_pe2.shape[0]),acc2)
        paint(pw1,pw2)


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