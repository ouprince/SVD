# -*- coding:utf-8 -*-
# SVD 矩阵分解程序
# 算法：梯度下降 + 权重正则

import sys, os
import numpy as np

class SVD(object):
    # UI 出现未知值的矩阵, 用 -1 表示未知数
    # K 表示分解矩阵 K 的维度 
    # learning_rate 学习率
    # lambda 正则化权重
    def __init__(self,UI,learning_rate = 0.01, lambdax = 0.1, K = 100):
        self.UI = UI # UI 矩阵，用 -1 代表 未知项
        self.learning_rate = learning_rate
        self.lambdax = lambdax
        self.K = K
        self.U, self.I = np.shape(UI)
        self.X = np.random.randn(self.U,self.K) # X 矩阵
        self.Y = np.random.randn(self.K,self.I) # Y 矩阵
        self.res = None
    
    def train(self,epochs = 10):
        for epoch in range(epochs):
            # 计算相乘矩阵
            R = np.dot(self.X,self.Y)
            # 计算误差矩阵 eui
            eui = self.UI - R
            SSE = 0
            for u in range(self.U):
                for i in range(self.I):
                    if UI[u,i] != -1:SSE+=eui[u,i] ** 2
            print("After %d epochs, The SSE = %.6f" %(epoch, SSE))
            if epoch > 0 and SSE > SSE_YUAN:
                self.learning_rate = 0.8 * self.learning_rate
            SSE_YUAN = SSE
            # 计算 puk 梯度 + 正则化 lambda , delta 为 X 的梯度矩阵
            delta_X = np.zeros((self.U,self.K))
            delta_Y = np.zeros((self.K,self.I))

            for u in range(self.U):
                for k in range(self.K): 
                    x = sum([eui[u,i] * self.Y[k,i] for i in range(self.I) if self.UI[u,i] != -1])/float(self.I)
                    y = self.lambdax * self.X[u,k]
                    delta_X[u,k] = (-x + y) * self.learning_rate 
        
            for k in range(self.K):
                for i in range(self.I):
                    x = sum([eui[u,i] * self.X[u,k] for u in range(self.U) if self.UI[u,i] != -1])/float(self.U)
                    y = self.lambdax * self.Y[k,i]
                    delta_Y[k,i] = (-x + y) * self.learning_rate

            # 梯度下降
            self.X = self.X - delta_X
            self.Y = self.Y - delta_Y
        self.res = np.dot(self.X,self.Y)
        
    def predict(self, user, item): # 预测 用户u 对 商品 i 的评价
        if self.res is None:
            raise BaseException("svd is not trained yet ... ")
        return self.res[user,item]
        

if __name__ == "__main__":
    # UI 矩阵，用 -1 表示 未知
    UI = np.array([[-1, 0, 0.5,0.8],[0.5,0.2,-1,0.1],[0.4,0.3,0.3,-1]])
    svd = SVD(UI)
    svd.train()
    print(svd.res)
