import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import copy
from parse import parse_arg
import numpy as np
import pandas as pd
import torch
import pickle
from model.Update import LocalUpdate
from model.Nets import MLP, CNNMnist, CNNCifar,DNN,LeNet
from model.Fed import FedAvg
from model.test import test_img
from reconstruct import reconstruct
import time
from sklearn.metrics import roc_auc_score

print(torch.cuda.is_available())

def split_data(data,ratio):
    np.random.seed(1)
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices =shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data[train_indices],data[test_indices]

if __name__ == '__main__':
    # parse args
    args=parse_arg()

    device = torch.device(args.device)
    # load dataset and split users
    if args.dataset == 'kdd':
        x=np.load(file="./data/kdd.npy",allow_pickle=True)
        print(x,x.shape)
        label=x[:,-1]
        train_data,test_data=split_data(data=x,ratio=0.3)
        data_num=len(train_data)//args.num_users
        discrete=[1,2,3,6,11,20,21]
        if args.test_single==1:
            test_data_temp=np.load(file="./data/kdd_full.npy",allow_pickle=True)
            test_label=test_data[:args.test_num,-1]
            test_x=test_data[:args.test_num,:-1]
            benign_idx=test_label==0
            temp_x=test_x[benign_idx]
            temp_label=test_label[benign_idx]
            temp_x=temp_x[:1000]
            temp_label=np.array(temp_label[:1000]).reshape(-1,1)
            for j in range(1,23):
                if j not in [4,5]:
                    train_data_j=train_data[train_data[:,-1]==j]
                    test_data_temp_j=test_data_temp[test_data_temp[:,-1]==j]
                    
                    for j1 in range(len(train_data_j)):
                        flag=0
                        for j2 in range(len(test_data_temp_j)):
                            if (train_data_j[j1]==test_data_temp_j[j2]).all()==True:
                                #print(j,"here")
                                flag=1
                                break
                        if flag==1:
                            test_data_temp_j=np.delete(test_data_temp_j,j2,axis=0)
                    
                    cur_x=test_data_temp_j[:1000,:-1]
                    cur_label=test_data_temp_j[:1000,-1].reshape(-1,1)
                    
                else:
                    cur_idx=test_label==j
                    cur_x=test_x[cur_idx]
                    cur_label=np.array(test_label[cur_idx]).reshape(-1,1)
                if j in [4,5]:
                    temp_x=np.vstack((temp_x,cur_x[:2000]))
                    temp_label=np.vstack((temp_label,cur_label[:2000]))
                else:
                    temp_x=np.vstack((temp_x,cur_x[:1000]))
                    temp_label=np.vstack((temp_label,cur_label[:1000]))
                print(j,len(cur_x))
            test_x=temp_x
            test_label=temp_label
            test_data=np.hstack((test_x,test_label))
            np.save("./data/kdd_test.npy",arr=test_data)
        #for i in range(23): print(np.sum(label==i))
        