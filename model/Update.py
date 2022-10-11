import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
import torch.nn.functional as F
from defense import defense,defense_cv,defense_instahide
from parse import parse_arg
args=parse_arg()
device = torch.device(args.device)

class DatasetSplit(Dataset):
    def __init__(self, dataset,args):
        self.dataset = dataset
        idx=self.dataset[:,-1]==0
        #self.dataset=self.dataset[idx]
        self.maxx=np.max(self.dataset[:,:-1],axis=0)
        self.minn=np.min(self.dataset[:,:-1],axis=0)
        self.dataset[:,:-1]=(self.dataset[:,:-1]-self.minn)/(self.maxx-self.minn+1e-9)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x = self.dataset[item,:-1]
        label=self.dataset[item,-1]
        return x, label
    
    def max_min(self):
        return self.maxx,self.minn

class LocalUpdate:
    def __init__(self, args, dataset,num_class):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        datasetsplit=DatasetSplit(dataset,args)
        if args.local_bs==-1: self.dataset = DataLoader(datasetsplit, batch_size=len(dataset), shuffle=True)
        else: self.dataset = DataLoader(datasetsplit, batch_size=self.args.local_bs, shuffle=True)
        self.maxx,self.minn=datasetsplit.max_min()
        self.ori_dataset=dataset
        
        self.num_class=num_class

    def train(self, net):
        #net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        
        epoch_loss = []
        batch_loss = []
        for batch_idx, (x, label) in enumerate(self.dataset):
            #print(batch_idx,x,label)
            optimizer.zero_grad()
            x = x.type(torch.FloatTensor)
            label = label.to(torch.int64)
            
            x=x.to(device)
            label=label.to(device)
            one_hot_int = F.one_hot(label, num_classes=self.num_class)
            one_hot=one_hot_int.float()
            log_probs = net(x)
            loss=-one_hot*torch.log(log_probs+1e-9)
            loss=torch.sum(loss,dim=1)
            loss=torch.mean(loss)
            
            if self.args.defense==1:
                #print("loss ori is ",loss.item())
                x_def,one_hot_def,diff=defense(copy.deepcopy(x),copy.deepcopy(one_hot),copy.deepcopy(net),self.args.defense_epochs,self.args.alpha)
                #optimizer.zero_grad()
                log_probs_def = net(x_def)
                loss1=-one_hot_def*torch.log(log_probs_def+1e-9)
                loss1=torch.sum(loss1,dim=1)
                loss1=torch.mean(loss1)
                loss1.backward()
                grad=[i.grad.detach() for i in net.parameters()]
                optimizer.step()
                
            elif self.args.defense==2:
                x_def=x
                one_hot_def=one_hot_int
                loss.backward()
                grad=[i.grad.detach() for i in net.parameters()]
                
                mask=defense_cv(copy.deepcopy(x),copy.deepcopy(one_hot),copy.deepcopy(net))
                #print("mask is"+str(mask)+str(len(mask)))
                #print(grad[2][0][:],grad[2].shape)
                grad[2] = grad[2] * torch.Tensor(mask).to(device)
                for i, p in enumerate(net.parameters()): p.grad=grad[i]
                #print(grad[2][0][:])
                optimizer.step()
                
            elif self.args.defense==3:
                x_def=x
                one_hot_def=one_hot_int
                loss.backward()
                grad=[i.grad.detach() for i in net.parameters()]
                threshold = [np.percentile(torch.abs(grad[i]).cpu().numpy(), 99) for i in range(len(grad))]
                
                threshold=np.asarray(threshold)
                
                threshold=torch.from_numpy(threshold).type(torch.FloatTensor).to(device)
                
                for i, p in enumerate(net.parameters()):
                    #print(p.grad,threshold[i])
                    p.grad[torch.abs(p.grad) < threshold[i]] = 0
                    #print(p.grad.shape)
                    
                grad=[i.grad.detach() for i in net.parameters()]
                #print(grad[1])
                optimizer.step()
                
            elif self.args.defense==4:
                x_def=x
                one_hot_def=one_hot_int
                loss.backward()
                grad=[i.grad.detach() for i in net.parameters()]
                #print("ori grad is",grad[0])

                for i, p in enumerate(net.parameters()):
                    #print("cur i gradient is",i)
                    #print(p.grad)
                    grad_tensor = p.grad.cpu().numpy()
                    delta=1e-1
                    #noise = np.random.laplace(0,delta, size=grad_tensor.shape)
                    noise = np.random.normal(0,delta, size=grad_tensor.shape)
                    #print(noise)
                    grad_tensor = grad_tensor + noise
                    p.grad = torch.Tensor(grad_tensor).to(device)
                    #print(p.grad)
                
                grad=[i.grad.detach() for i in net.parameters()]
                #print("after is ",grad[0])
                #print(grad[1])
                optimizer.step()
                
            elif self.args.defense==5:
                
                x_def,one_hot_def=defense_instahide(self.args,self.ori_dataset,self.num_class)
                #print("loss def is",diff.item())
                #print(x_def,x_def.shape)

                optimizer.zero_grad()
                log_probs_def = net(x_def)
            
                loss1=-one_hot_def*torch.log(log_probs_def+1e-9)
                loss1=torch.sum(loss1,dim=1)
                loss1=torch.mean(loss1)

                loss1.backward()
                grad=[i.grad.detach() for i in net.parameters()]
                #print("after is ",grad[0])
                optimizer.step()
             
            else : 
                x_def=x
                one_hot_def=one_hot_int
                loss.backward()
                grad=[i.grad.detach() for i in net.parameters()]
                optimizer.step()

            #print("after is ",grad[2],torch.sum(grad[2]!=0))
            
            batch_loss.append(loss.item()) 
            if args.attack!=0: break
            break
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), grad, x, one_hot_int,x_def,one_hot_def,self.maxx,self.minn

