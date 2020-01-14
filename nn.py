import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
class NeuralNet:
    def __init__(self,hidden_layers,hidden_units,random_state=5,learning_rate=0.05,n_epochs=10000,initialization='random',
                 Regularization='L1',lamda=0):
        self.learning_rate,self.n_epochs,self.Regularization,self.lamda,self.lss = learning_rate,n_epochs,Regularization,lamda,[]
        self.hidden_layers,self.hidden_units,self.reg_term = hidden_layers,hidden_units,None
        self.random_state = random_state
        self.initialization = initialization
    def net_initlze(self,x,y):
        arch = [{'input' : x.shape[1]}]
        for i in range(self.hidden_layers):
            arch.append({'n_units':self.hidden_units})
        arch.append({'output':y.shape[1]})
        random.seed(self.random_state)
        neuralNetwork = {}
        if self.initialization=='random':
            neuralNetwork['w1'] = random.rand(arch[0]['input'],arch[1]['n_units'])
            neuralNetwork['b1'] = random.rand(arch[1]['n_units'])
            if len(arch) > 3:
                for i in range(2,len(arch)-1):
                    neuralNetwork['w'+str(i)] = random.rand(arch[i-1]['n_units'],arch[i]['n_units'])
                    neuralNetwork['b'+str(i)] = random.rand(arch[i]['n_units'])

            neuralNetwork['w'+str(len(arch)-1)] = random.rand(arch[-2]['n_units'],arch[-1]['output'])
            neuralNetwork['b' +str(len(arch)-1)] = random.rand(arch[-1]['output'])
        if self.initialization == 'same' :
            neuralNetwork['w1'] = np.ones((arch[0]['input'],arch[1]['n_units']))
            neuralNetwork['b1'] = np.ones(arch[1]['n_units'])
            if len(arch) > 3:
                for i in range(2,len(arch)-1):
                    neuralNetwork['w'+str(i)] = np.ones((arch[i-1]['n_units'],arch[i]['n_units']))
                    neuralNetwork['b'+str(i)] = np.ones(arch[i]['n_units'])
            neuralNetwork['w'+str(len(arch)-1)] = np.ones((arch[-2]['n_units'],arch[-1]['output']))
            neuralNetwork['b' +str(len(arch)-1)] = np.ones(arch[-1]['output'])
        return neuralNetwork
    def ForwardPropagation(self,network,inpt):
        mmry = {}
        mmry['a0'] = inpt
        ln = int(len(self.neuralNetwork)/2)
        for i in range(1,ln): #len(arch)-1
            idx = str(i)
            w,b = 'w'+idx,'b'+idx
            wx_b = inpt.dot(network[w]) + network[b]
            inpt = 1 / (1 + np.exp(-wx_b))
            mmry['a'+idx],mmry['z'+idx] = inpt,wx_b
        idx = str(ln)
        w,b = 'w'+ idx ,'b'+ idx
        wx_b = inpt.dot(network[w]) + network[b]
        inpt = np.exp(wx_b) / np.sum(np.exp(wx_b),axis=1,keepdims=1)
        mmry['a'+idx],mmry['z'+idx] = inpt,wx_b
        return inpt,mmry
    def actv_derivative(self,inpt):
        sig = 1 / (1 + np.exp(-inpt))
        return sig * (1 - sig)
    def d_w_b(self,dz,a,m):
        dw = dz.T.dot(a)/m
        db = dz.sum(axis=0)/m
        return dw,db
    def loss(self,ytrue,ypred):
        return np.mean(-np.sum(ytrue*np.log(ypred),axis=1))
    def Reg_value(self,ln,network):
        Regularization = 0
        if self.Regularization=='L2' :
            for i in range(ln): Regularization+= np.sum(network['w'+str(i+1)])
        if self.Regularization=='L1' :
            for i in range(ln): Regularization+=np.sum((network['w1']>0)*1 + (network['w1']<0)*-1)
        return Regularization
    
    def backPropagation(self,network,mmry ,inpt,ytrue):
        ln = int(len(network)/2)
        n_lyrs = ln
        updates={}
        self.reg_term = self.Reg_value(ln,network)
        cost = mmry['a'+str(n_lyrs)] - ytrue + self.lamda * self.reg_term * (1/ytrue.shape[0])
        da_last = cost.dot(network['w'+str(n_lyrs)].T)
        m = ytrue.shape[0]
        idx = str(n_lyrs-1)
        dw,db = self.d_w_b(cost,mmry['a'+idx],m)    
        updates['w'+str(n_lyrs)],updates['b'+str(n_lyrs)] = dw,db
        i = ln-1
        while i>=1:
            dz = da_last * self.actv_derivative(mmry['z'+str(i)])
            dw,db = self.d_w_b(dz,mmry['a'+str(i-1)],m)
            updates['w'+str(i)],updates['b'+str(i)] = dw,db
            if i >1 : da_last = da_last.dot(network['w'+str(i)].T)
            i-=1
        return updates
    def network_update(self,network,updates,learning_rate,n):
        for i in range(int(len(network)/2)):
            idx = str(i+1)
            network['w'+idx] -= learning_rate * updates['w'+idx].T+self.lamda*self.reg_term
            network['b'+idx] -= learning_rate * updates['b'+idx].T
        
    def fit(self,x,y):
        y = pd.get_dummies(y).values
        n = y.shape[0]
        self.neuralNetwork = self.net_initlze(x,y)
        x = (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))
        for i in range(self.n_epochs):
            ypred,mm = self.ForwardPropagation(self.neuralNetwork,x)
            updts = self.backPropagation(self.neuralNetwork,mm,x,y)
            self.network_update(self.neuralNetwork,updts,self.learning_rate,n)
            self.lss.append(self.loss(y,ypred))
    def predict(self,x):
        x = (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))
        return self.ForwardPropagation(self.neuralNetwork,x)[0]
    def loss_histry(self):
        plt.plot(self.lss)
        plt.xlabel('Epochs')
        plt.xlabel('Loss')
