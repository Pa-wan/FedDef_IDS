import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
import torch.nn.functional as F
from defense import defense,defense_cv,defense_instahide

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate:
    def __init__(self, args, dataset,num_class):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset=copy.deepcopy(dataset)
        self.maxx=np.max(self.dataset[:,:-1],axis=0)
        self.minn=np.min(self.dataset[:,:-1],axis=0)
        self.dataset[:,:-1]=(self.dataset[:,:-1]-self.minn)/(self.maxx-self.minn+1e-9)
        
        idx=self.dataset[:,-1]==0
        #self.dataset=self.dataset[idx]
        
        self.num_class=num_class

    def train(self, net):
        #net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            np.random.shuffle(self.dataset)
            data=self.dataset[:self.args.local_bs]
            batch_loss = []
            x=data[:,:-1]
            label=data[:,-1]
            
            
            x = torch.from_numpy(x).type(torch.FloatTensor) 
            label = torch.from_numpy(label).to(torch.int64)
 
            one_hot_int = F.one_hot(label, num_classes=self.num_class)
            
            one_hot=one_hot_int.float()
            log_probs = net(x)
            loss=-one_hot*torch.log(log_probs+1e-9)
            loss=torch.sum(loss,dim=1)
            loss=torch.mean(loss)
            
            #print(x,label,one_hot)
            
            if self.args.defense==1:
                #print("loss ori is ",loss.item())
                x_def,one_hot_def,diff=defense(copy.deepcopy(x),copy.deepcopy(one_hot),net,
                                          self.args.local_bs*self.args.recon_epochs,self.args.defense_epochs,self.args.alpha)
                print("loss def is",diff.item())
                #print(x_def,one_hot_def)
                

                optimizer.zero_grad()
                log_probs_def = net(x_def)
                #print(log_probs_def)
            
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
                print("mask is"+str(mask)+str(len(mask)))
                #print(grad[2][0][:],grad[2].shape)
                grad[2] = grad[2] * torch.Tensor(mask)
                for i, p in enumerate(net.parameters()): p.grad=grad[i]
                #print(grad[2][0][:])
                optimizer.step()
                
            elif self.args.defense==3:
                x_def=x
                one_hot_def=one_hot_int
                loss.backward()
                grad=[i.grad.detach() for i in net.parameters()]
                threshold = [torch.quantile(torch.abs(grad[i]), 0.99) for i in range(len(grad))]

                
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
                    grad_tensor = p.grad.numpy()
                    noise = np.random.laplace(0,1e-1, size=grad_tensor.shape)
                    #noise = np.random.normal(0,1e-1, size=grad_tensor.shape)
                    #print(noise)
                    grad_tensor = grad_tensor + noise
                    p.grad = torch.Tensor(grad_tensor)
                    #print(p.grad)
                
                grad=[i.grad.detach() for i in net.parameters()]
                #print("after is ",grad[0])
                #print(grad[1])
                optimizer.step()
                
            elif self.args.defense==5:
                
                x_def,one_hot_def=defense_instahide(self.args,self.dataset,self.num_class)
                #print("loss def is",diff.item())

                optimizer.zero_grad()
                log_probs_def = net(x_def)
            
                loss1=-one_hot_def*torch.log(log_probs_def+1e-9)
                loss1=torch.sum(loss1,dim=1)
                loss1=torch.mean(loss1)

                loss1.backward()
                grad=[i.grad.detach() for i in net.parameters()]
                #print("after is ",grad[0])
                optimizer.step()
                
            elif self.args.defense==6:
                
                x_def,one_hot_def,diff=defense_6(copy.deepcopy(x),copy.deepcopy(one_hot),copy.deepcopy(net),
                                          self.args.local_bs*self.args.recon_epochs,self.args.defense_epochs,self.args.alpha)
                #print("loss def is",diff.item())

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

            print("after is ",grad[1],torch.sum(grad[1]!=0))
            batch_loss.append(loss.item())    
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), grad, x, one_hot_int,x_def,one_hot_def,self.maxx,self.minn

