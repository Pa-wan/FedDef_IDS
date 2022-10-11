import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
import torch.nn.functional as F
from direct_gradient import direct_gradient
from parse import parse_arg
args=parse_arg()
device = torch.device(args.device)

def transform(x,y,num_class,maxx,minn,discrete):
    x=torch.clamp(x, min=0,max=1)
    index=torch.max(y,1)[1]
    y = F.one_hot(index, num_classes=num_class).float()
    
    if len(discrete)==0: return x.detach(),y.detach()
    
    x=x*(maxx-minn+1e-9)+minn
    
    x[:,discrete]=torch.round(x[:,discrete])
    
    x=(x-minn)/(maxx-minn+1e-9)
    
    return x.detach(),y.detach()

def trans(x):
    return (torch.tanh(x)+1)/2

def reconstruct(attack,model,grad,batch,feature_size,num_class,epochs,maxx,minn,discrete):
    
    x=torch.rand((batch,feature_size)).to(device).requires_grad_(True)
    y=torch.rand((batch,num_class)).to(device).requires_grad_(True)
    lr=3e-2
    
    optimizer = torch.optim.Adam([x,y], lr=lr)
    
    for i in range(epochs):
        def closure():
            x.requires_grad=True
            y.requires_grad=True
            optimizer.zero_grad()
        
            log_probs=model(x)
        
            #print(log_probs)
        
            model_loss=-y*torch.log(log_probs+1e-9)
            model_loss=torch.sum(model_loss,dim=1)
            model_loss=torch.mean(model_loss)

            cur_grad=torch.autograd.grad(model_loss, model.parameters(), create_graph=True)
        
            loss=0

            for j in range(len(grad)): 
                #loss+=1-torch.cosine_similarity(grad[j].flatten(),cur_grad[j].flatten(),dim=0)
                loss+=torch.sqrt(torch.mean((grad[j]-cur_grad[j])**2)+1e-9)
            
            loss/=len(grad)
            loss.backward()
            return loss
        
        #print(x.grad,y.grad,x,y)
        loss=closure()
        
        x_temp=copy.deepcopy(x)
        y_temp=copy.deepcopy(y)
        optimizer.step()
        
        epoch_per_batch=epochs//batch
        
        cur_batch=(i+1)//epoch_per_batch+1
        
        if cur_batch<batch: 
            x.detach_()
            y.detach_()
            #print("here")
            x[cur_batch:]=x_temp[cur_batch:]
            y[cur_batch:]=y_temp[cur_batch:]
            x.detach_()
            y.detach_()

        if loss.item()<1e-5: break
        #print("cur i is ",i,"and loss is",loss.item())#,"and x is",x.detach().numpy(),"and y is",y.detach().numpy())
        
    if attack==2: 
        flag,x_direct=direct_gradient(grad[0],grad[1])
        if flag==0: 
            x=x_direct
            x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        else: print("extraction failed")
        
    return transform(x,y,num_class,torch.from_numpy(maxx).type(torch.FloatTensor).to(device),
                     torch.from_numpy(minn).type(torch.FloatTensor).to(device),discrete)
    
        
    
    
    