import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
import torch.nn.functional as F
from parse import parse_arg
args=parse_arg()
device = torch.device(args.device)

def direct_gradient(dw,db_ori):
    
    db=torch.zeros((1,len(db_ori))).to(device)
    
    for i in range(db.shape[0]):
        db[i]=db_ori
        
    #print(db)
    
    temp1=torch.mm(dw.T,db.T)
    
    flag=0
    
    temp2=torch.mm(db,db.T)
    
    if temp2==0:
        flag=1
        return flag,temp2
    
    temp3=torch.inverse(temp2)
    
    return flag,torch.mm(temp1,temp3).T