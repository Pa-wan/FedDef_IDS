import sys
import os
sys.path.append('./')
import numpy as np
import corClust as CC
import pickle
import torch.nn as nn
import torch
import torch.optim as optim
from Autoencoder import Net
import copy

print(torch.cuda.is_available())


LOSS_FUNC = nn.MSELoss()    # 损失函数，train时用，loss返回标量
LOSS_CALC = nn.MSELoss(reduction='none')		# 损失函数，调用exeucte_torch时用，loss返回向量

AD_RATE = 0.9   # 选择阈值时，将训练集中的全部rmse递增排序，选择AD_rate位置的值

# 将nn.MSELoss(reduction='none')的结果转化成kitsune中RMSE
def se2rmse(a):
    return torch.sqrt(sum(a.t())/a.shape[1])

class KitNET:

    # n : feature
    def __init__(self, feature_size, max_autoencoder_size=10, FM_grace_period=None, AD_grace_period=10000,
                 learning_rate=0.05, hidden_ratio=0.75, feature_map=None):

        # Parameters:
        self.m = max_autoencoder_size
        self.n = feature_size
        self.AD_grace_period = AD_grace_period
        self.FM_grace_period = FM_grace_period
        self.lr = learning_rate
        self.hr = hidden_ratio

        # Variables:
        self.n_trained = 0  # the number of training instances so far
        self.RMSEs = []

        # Model:
        self.v = feature_map
        self.FM = CC.corClust(self.n)  # incremental feature cluatering for the feature mapping process
        self.ensembleLayer = []
        self.norm_max = []
        self.norm_min = []
        self.outputLayer = None
        self.ad_threshold = -1. 
      
    def subtrain(self,x,label=0):
        # ----- Ensemble Layer ------------------ +
        Loss = torch.zeros(len(self.ensembleLayer)) # Ensemble layer 的输出数组

        pgd=0
        for i in range(len(self.ensembleLayer)):
            xi = x[self.v[i]]
            if (label==0):
                self.norm_max[i][xi > self.norm_max[i]] = xi[xi > self.norm_max[i]]
                self.norm_min[i][xi < self.norm_min[i]] = xi[xi < self.norm_min[i]]
            # 0-1 normalize
            xi = (xi - self.norm_min[i]) / (self.norm_max[i] - self.norm_min[i] + 0.0000000000000001)
            
            # create optimizer
            optimizer = optim.SGD(self.ensembleLayer[i].parameters(), lr=self.lr)
            
            optimizer.zero_grad()  # zero the gradient buffers
            output = self.ensembleLayer[i](xi)
            #测试这里去掉根号的loss
            #loss = torch.sqrt(LOSS_FUNC(output, xi))
            loss = LOSS_FUNC(output, xi)
            if (label==1): 
                loss=torch.relu((4-torch.sqrt(loss))**2)
            loss.backward()
            
            optimizer.step()  # Does the update
            # 更新参数之后再求一次RMSE,作为Ensemble层的输出
            output = self.ensembleLayer[i](xi)
            Loss[i] = torch.sqrt(LOSS_FUNC(output, xi))
            #print("i is:"+str(i)+" loss is:"+str(Loss[i]))

        # ----- OutputLayer ------------------- +
        # create optimizer   
        i=len(self.ensembleLayer)
        if (label==0):
            self.norm_max[i][Loss > self.norm_max[i]] = Loss[Loss > self.norm_max[i]]
            self.norm_min[i][Loss < self.norm_min[i]] = Loss[Loss < self.norm_min[i]]
        # 0-1 normalize
        #print(self.norm_max[i],self.norm_min[i])
        Loss = (Loss - self.norm_min[i]) / (self.norm_max[i] - self.norm_min[i] + 0.0000000000000001)
        Loss.detach_()
        optimizer = optim.SGD(self.outputLayer.parameters(), lr=self.lr)

        optimizer.zero_grad()  # zero the gradient buffers
        output = self.outputLayer(Loss)
        #loss = torch.sqrt(LOSS_FUNC(output, Loss))
        loss = LOSS_FUNC(output, Loss)
        loss_data=np.sqrt(loss.item())
        #print(torch.sqrt(loss))
        if (label==1): 
            loss=torch.relu((4-torch.sqrt(loss))**2)
        loss.backward()
       
        optimizer.step()  # Does the update

        if (self.n_trained%1000==0):
            print("now is :"+str(self.n_trained)+"final loss is :"+str(loss_data))
        return loss_data
        #print(self.RMSEs)
    
    def train(self, x,label=0):
        ## train FM
        if self.n_trained <= self.FM_grace_period:
            # update the incremental correlation matrix
            self.FM.update(x)

            if self.n_trained == self.FM_grace_period: #If the feature mapping should be instantiated
                self.v = self.FM.cluster(self.m)
                

                self.__buildKitNETModel__()
                print("|-- The Feature-Mapper found a mapping: "+str(self.n)+" features to "+str(len(self.v))+" autoencoders.")

        ## train KitNET
        else:
            x = torch.from_numpy(x).type(torch.FloatTensor) 
            rmse=self.subtrain(x,label)
            self.RMSEs.append(rmse)

        self.n_trained += 1
        # train阶段结束:
        if self.n_trained == self.AD_grace_period+self.FM_grace_period-1:
            # 排序RMSE并求阈值
            self.RMSEs.sort()
            # self.RMSEs = np.asarray(self.RMSEs)/len(self.v)
            # self.RMSEs = np.sqrt(self.RMSEs)
            with open('./__train_rmse__.pkl', 'wb') as f:
                pickle.dump(self.RMSEs, f)
            self.ad_threshold = self.RMSEs[int(len(self.RMSEs)*AD_RATE)-1]
            print("|-- ad_threshold:", self.ad_threshold)
    
    def norm(self,x):
        #print("x is"+str(x))
        for i in range(len(self.ensembleLayer)):
            xi = x[:,self.v[i]]
            #print("self.vi is"+str(self.v[i]))
            #print("xi is"+str(xi))
            # 0-1 normalize
            #print(self.norm_min[i],self.norm_max[i])
            xi = (xi - self.norm_min[i]) / (self.norm_max[i] - self.norm_min[i] + 0.0000000000000001)
            x[:,self.v[i]]=xi
        return x
    
    def unnorm(self,x):
        for i in range(len(self.ensembleLayer)):
            #print(self.v[i],self.norm_min[i],self.norm_max[i])
            xi = x[:,self.v[i]]
            xi=xi*(self.norm_max[i] - self.norm_min[i] + 0.0000000000000001)+self.norm_min[i]
            x[:,self.v[i]]=xi
        return x
    
    def subexecute(self,x):
        # ----- Ensemble Layer ------------------ +
        Loss = torch.zeros(len(x),len(self.ensembleLayer))
        for i in range(len(self.ensembleLayer)):
            xi = x[:,self.v[i]]
            # 0-1 normalize
            
            output = self.ensembleLayer[i](xi)
            loss = LOSS_CALC(xi,output)
            loss=torch.mean(loss,dim=1)
            Loss[:,i]=torch.sqrt(loss)
        # 0-1 normalize
        i=len(self.ensembleLayer)
        # 0-1 normalize
        Loss = (Loss - self.norm_min[i]) / (self.norm_max[i] - self.norm_min[i] + 0.0000000000000001)
        # ----- OutputLayer ------------------- +
        output = self.outputLayer(Loss)
        loss = LOSS_CALC(Loss,output)
        loss=torch.mean(loss,dim=1)
        return loss

    def execute(self,x):
        pgd=0
        self.ad_threshold = self.RMSEs[int(len(self.RMSEs)*AD_RATE)-1]
        #x = torch.from_numpy(x).type(torch.FloatTensor)
        #x=self.norm(x)
        #print("after norm",torch.max(x))
        #if pgd==1: x=self.ae(x,label)
        rmse=self.subexecute(x)
        rmse=torch.sqrt(rmse)
        return rmse.detach().numpy()
    
    def transform(self,x):
        x=torch.clamp(x, min=0)
        x=self.unnorm(x)
        idx=[1,2,3,6,11,20,21]
        #idx=[0,1,2,3,4,5]
        for cur in idx: 
            x[:,cur]=torch.round(x[:,cur])
            if cur not in [1,2,3]:
                x[:,cur]=torch.clamp(x[:,cur],min=0,max=1)
        x=self.norm(x)
        return x
    
    def ae(self,xi,label):
        with open("./adv/__model__.pkl", 'rb') as f:   # 读取模型信息
             net = pickle.load(f)
        print("net2 is"+str(net))
        ori=xi.numpy()
        
        label = np.array(label, dtype = int)
        rd = np.random.RandomState(888)
        rarr=[]
        step=2/255*1
        constraint=0.075*30
        
        x_adv=copy.deepcopy(xi)#+xi*torch.Tensor(rarr)

        for k in range(200):
            x_adv.requires_grad=True

            loss=self.subexecute(x_adv)
            #loss+=net.execute(x_adv,label)
            
            bw=(x_adv[:,4]+x_adv[:,5])/(x_adv[:,0]+1e-0)
            #bw=x_adv[:,19]+(x_adv[:,9]+x_adv[:,10])/(x_adv[:,6]+1)
            idx=[22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
            #bw+=torch.cosine_similarity(xi[:,idx],x_adv[:,idx],dim=1)

            bw=0.01*bw
            
            
            loss-=bw
            #print(bw)
            #loss+=1*torch.sqrt(torch.mean((x_adv-xi)**2,dim=1)+1e-9)
            temp=torch.ones((len(x_adv)))
            #print("now is "+str(k+1)+str(loss[:])+str(len(loss)))
            print("now is "+str(k+1)+" and loss is "+str(torch.sqrt(loss[:]+bw)))
            loss.backward(temp)
            
            temp=torch.ones((x_adv.shape[0],x_adv.shape[1]))
            temp[label==1,:]=-1
            temp[label==0,:]=1
            
            L_x=x_adv.grad        
            x_adv = x_adv + temp*step*torch.sign(L_x)

            eta = torch.clamp(x_adv-xi, min=-constraint, max=constraint)
            #print(eta)
            x_adv=xi+eta
            x_adv=torch.clamp(x_adv, min=0)
            if k==200-1: x_adv=self.transform(x_adv)
            #idx=[1,2,3,6,11,20,21]
            #for it in idx:
            #    print("cur adv is "+str(x_adv[:,it]))
                
            
            x_adv.detach_()
        print(ori[:,0:6],x_adv[:,0:6])
        
        temp=copy.deepcopy(xi)
        temp=self.unnorm(temp)
        ori_bw=(temp[:,4]+temp[:,5])/(temp[:,0]+1e-0)
        print("now is ori form :\n"+str(temp[:,0:6]))
        
        temp=copy.deepcopy(x_adv)
        temp=self.unnorm(temp)
        adv_bw=(temp[:,4]+temp[:,5])/(temp[:,0]+1e-0)
        print("now is adv ori form :\n"+str(temp[:,0:6]))
        idx=[1,2,3,6,11,20,21]
        #for it in idx:
        #    print("cur adv is "+str(temp[:,it]))
        
        print("now is bw :\n")
        print(ori_bw,adv_bw)
        
        ori=torch.from_numpy(ori).type(torch.FloatTensor) 
        temp=torch.sqrt(torch.mean((x_adv-ori)**2,dim=1))
        print("diff is"+str(temp[:]))
        return x_adv

    def __buildKitNETModel__(self):
        # construct ensemble layer
        for map in self.v:
            net = Net(len(map), self.hr)
            # net.cuda()
            self.ensembleLayer.append(net)
            self.norm_max.append(torch.ones((len(map),)) * -np.Inf)
            self.norm_min.append(torch.ones((len(map),)) * np.Inf)

        # construct output layer
        self.outputLayer = Net(len(self.v), self.hr)
        self.norm_max.append(torch.ones((len(self.v),)) * -np.Inf)
        self.norm_min.append(torch.ones((len(self.v),)) * np.Inf)
        # self.outputLayer.cuda()