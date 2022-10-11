import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from torch.utils.data import DataLoader, Dataset
import copy
from parse import parse_arg
import numpy as np
import pandas as pd
import torch
import pickle
from model.Update import LocalUpdate,DatasetSplit
from model.Nets import *
from model.densenet import *
from model.resnet import *
from model.Fed import FedAvg
from reconstruct import reconstruct
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import time
from sklearn.metrics import roc_auc_score
import KitNET

print(torch.cuda.is_available())

def split_data(data,ratio):
    np.random.seed(1)
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices =shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data[train_indices],data[test_indices]

def get_key(values):
    value_cnt = {}  # 将结果用一个字典存储
    # 统计结果
    for value in values: value_cnt[value] = value_cnt.get(value, 0) + 1
    # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
    return list(value_cnt.keys())

def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cal_score(ori_x,recon_x,maxx,minn,discrete):
    recon_x_rev=recon_x*(maxx-minn+1e-9)+minn
    ori_x_rev=ori_x*(maxx-minn+1e-9)+minn

    #print(ori_x_rev,recon_x_rev)
    
    if len(discrete)!=0: 
        ori_dis=ori_x_rev[:,discrete]
        recon_dis=recon_x_rev[:,discrete]
        score=torch.sum(ori_dis!=recon_dis,dim=1)
    else: score=torch.zeros((ori_x.shape[0],))
    #print(score)
    score=score.float()
    for i in range(ori_x.shape[1]):
        #print(ori_x[:,i],recon_x[:,i],score)
        if i not in discrete: score+=abs((ori_x[:,i]-recon_x[:,i]))
    return score/ori_x.shape[1]

def generate_non(data,args,num_class,user_id):
    np.random.seed(user_id)
    num_attack=np.random.randint(low=1,high=num_class)
    #if args.dataset=='kdd': num_attack=np.random.randint(low=1,high=num_class)
    #else: num_attack=np.random.randint(low=1,high=6)
    idx=data[:,-1]==0
    benign_data=data[idx]
    malicious_data=data[~idx]
    np.random.shuffle(benign_data)
    result=benign_data[:len(benign_data)//args.num_users]
    
    prob1=[]
    all_class=range(1,num_class)
    for i in range(1,num_class):
        idx=data[:,-1]==i
        temp=np.sum(idx)/len(malicious_data)
        prob1.append(temp)

    if num_attack>=2:
        #attack_type_1=np.random.choice(a=all_class, size=2, replace=False,p=prob1)
        attack_type_1=np.random.choice(a=all_class, size=2, replace=False)
        attack_type_2=np.random.choice(a=all_class, size=num_attack-2, replace=False)
        attack_type=np.hstack((attack_type_1,attack_type_2))
    else: attack_type=np.random.choice(a=all_class, size=num_attack, replace=False)

    for i in attack_type:
        idx=data[:,-1]==i
        malicious_data=data[idx]
        np.random.shuffle(malicious_data)
        result=np.vstack((result,malicious_data[:len(malicious_data)//args.num_users+100]))
    return result

def compute_pos_neg(y_actual, y_hat):
    TP = 0; FP = 0;TN = 0; FN = 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1: TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]: FP += 1
        if y_actual[i]==y_hat[i]==0: TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]: FN += 1
    return TP,FP,TN,FN

def metrics(TP,FP,TN,FN):
    a=TP+FP
    b=TP+FN
    c=TN+FP
    d=TN+FN
    #mcc=((TP*TN)-(FP*FN))/(math.sqrt(float(a*b*c*d)+0.0001))
    F1=(2*TP)/float(2*TP+FP+FN+.0000001)
    precision=TP/float(TP+FP+.0000001)
    recall=TP/float(TP+FN+.0000001)
    TPR=TP/(TP+FN)
    FPR=FP/(FP+TN)
    return TPR,FPR,F1,precision,recall

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
        print(train_data.shape,test_data.shape)
        data_num=len(train_data)//args.num_users
        discrete=[1,2,3,6,11,20,21]
        #for i in range(23): print(np.sum(label==i))
    
    if args.dataset == 'mirai':
        x=np.load(file="./data/mirai.npy",allow_pickle=True)
        print(x,x.shape)
        label=x[:,-1]
        train_data,test_data=split_data(data=x,ratio=0.3)
        print(train_data.shape,test_data.shape)
        data_num=len(train_data)//args.num_users
        discrete=[]
        
    if args.dataset == 'cic2017':
        x=np.load(file="./data/cic2017.npy",allow_pickle=True)
        print(x.shape)
        label=x[:,-1]
        train_data,test_data=split_data(data=x,ratio=0.3)
        print(train_data.shape,test_data.shape)
        data_num=len(train_data)//args.num_users
        discrete=[]
    
    if args.dataset == 'unsw':
        train_data=np.load(file="./data/unsw_train.npy",allow_pickle=True)
        test_data=np.load(file="./data/unsw_test.npy",allow_pickle=True)
        x=np.vstack((train_data,test_data))
        print(x.shape)
        label=x[:,-1]
        train_data,test_data=split_data(data=x,ratio=0.3)
        print(train_data.shape,test_data.shape)
        data_num=len(train_data)//args.num_users
        discrete=[]
        
    alpha=int(args.alpha*100)
    
    # build model
    if args.model == 'dnn':
        num_class=get_key(label)
        print(len(num_class))
        net_glob = DNN(x.shape[1]-1,len(num_class))

    if args.load_model==1: 
        if args.defense!=1:
            with open('./save_model/{}_{}_{}_model.pkl'.format(args.dataset,args.model,args.defense), 'rb') as f:  net_glob = pickle.load(f)
        else:
            with open('./save_model/{}_{}_{}_alpha_{}_model.pkl'.format(
                args.dataset,args.model,args.defense,alpha), 'rb') as f: net_glob=pickle.load(f)
                
    net_glob.to(device)
    print(net_glob,len(list(net_glob.parameters())))
    #if args.dataset=='cifar': net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training

    print("Aggregation over all clients")
    w_locals = [w_glob for i in range(args.num_users)]
    data_print = open("TrainTimeRecord.txt",'w',encoding="utf-8")
    
    score1_plot=[]
    score2_plot=[]
    train_loss_plot = []
    test_acc_plot=[]
    
    recon_x_save=[]
    recon_y_save=[]
    
    #if args.defense!=1:
    #    recon_x_save=list(np.load('./gan_dataset/x_{}_{}_{}.npy'.format(args.dataset, args.model, args.defense)))
    #    recon_y_save=list(np.load('./gan_dataset/y_{}_{}_{}.npy'.format(args.dataset, args.model, args.defense)))
    #else:
    #    recon_x_save=list(np.load('./gan_dataset/x_{}_{}_{}_alpha_{}.npy'.format(args.dataset, args.model, args.defense,alpha)))
    #    recon_y_save=list(np.load('./gan_dataset/y_{}_{}_{}_alpha_{}.npy'.format(args.dataset, args.model, args.defense,alpha)))
    
    if args.iid==0:
        result_data=[]
        for i in range(args.num_users):
            data_non=generate_non(train_data,args,len(num_class),i)
            result_data.append(data_non)
            print("User ",i+1,": ",np.sort(get_key(data_non[:,-1])))
            
    start=time.time()
    
    for cur_i in range(args.epochs):
        if (cur_i+1)%args.decay_epochs==0:
            args.lr=args.lr*args.decay
            print(args.lr)
            print(args.lr,file=data_print)
        
        loss_locals = []
        for i in range(args.num_users):
            if args.dataset in ['cic2017','mirai'] or args.iid==1 :
                local = LocalUpdate(args=args, dataset=copy.deepcopy(train_data[i*data_num:(i+1)*data_num]),num_class=len(num_class)) 
            else: local = LocalUpdate(args=args, dataset=copy.deepcopy(result_data[i]),num_class=len(num_class))
  
            w, loss, grad, train_x,train_one_hot,train_x_def,train_one_hot_def,train_max,train_min = local.train(
                net=copy.deepcopy(net_glob).to(device))
            w_locals[i] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
            
            #print("cur user is ",i,"and loss is ",loss)
            if i!=0 or args.attack==0: continue
            #reconstruct gradients
            #print(grad[1],grad[1].shape)
            
            if args.save_change==1:
                args_temp=copy.deepcopy(args)
                args_temp.local_bs=1
                local = LocalUpdate(args=args_temp, dataset=train_data[i*data_num:(i+1)*data_num],num_class=len(num_class))
                w, loss, grad, train_x,train_one_hot,train_x_def,train_one_hot_def,train_max,train_min = local.train(net=copy.deepcopy(net_glob))
                recon_x,recon_y=reconstruct(args_temp.attack,net_glob,grad,args_temp.local_bs,x.shape[1]-1,len(num_class),
                                        args_temp.local_bs*args_temp.recon_epochs,train_max,train_min,discrete)

            else: recon_x,recon_y=reconstruct(args.attack,net_glob,grad,args.local_bs,x.shape[1]-1,len(num_class),
                                        args.local_bs*args.recon_epochs,train_max,train_min,discrete)
            
            train_x=train_x.cpu()
            train_x_def=train_x_def.cpu()
            recon_x=recon_x.cpu()
            train_one_hot=train_one_hot.cpu()
            train_one_hot_def=train_one_hot_def.cpu()
            recon_y=recon_y.cpu()

            score1=cal_score(train_x,recon_x,torch.from_numpy(train_max).type(torch.FloatTensor),
                             torch.from_numpy(train_min).type(torch.FloatTensor),discrete)
            if torch.any(torch.isnan(score1)): print(recon_x)
            #score1=torch.sqrt(torch.mean((train_x-recon_x)**2,dim=1)+1e-9)
            score2=torch.max(recon_y,1)[1]==torch.max(train_one_hot,1)[1]
            score2=torch.sum(score2)/score1.shape[0]

            if args.save_recon==1:
                for recon_i in range(args.local_bs):
                    recon_x_save.append(recon_x[recon_i].numpy())
                    recon_y_save.append(recon_y[recon_i].numpy())
                if args.defense!=1:
                    np.save('./gan_dataset/x_{}_{}_{}.npy'.format(args.dataset, args.model, args.defense),arr=recon_x_save)
                    np.save('./gan_dataset/y_{}_{}_{}.npy'.format(args.dataset, args.model, args.defense),arr=recon_y_save)
                else:
                    np.save('./gan_dataset/x_{}_{}_{}_alpha_{}.npy'.format(args.dataset, args.model, args.defense,alpha),arr=recon_x_save)
                    np.save('./gan_dataset/y_{}_{}_{}_alpha_{}.npy'.format(args.dataset, args.model, args.defense,alpha),arr=recon_y_save)
            
            if i==0:
                score1_plot.append(score1.numpy())
                score2_plot.append(score2.numpy())
            print("and score 1 is",score1.numpy(),"and score 2 is",score2.numpy(),"and true label is",train_one_hot.numpy())
            #print("and score 1 is",score1.numpy(),"and score 2 is",score2.numpy(),"and train_x is\n",train_x.numpy(),"\n\nand recon_x is\n",recon_x.numpy(),"\n\nand train_x_def is\n",train_x_def.numpy(),"and train_y is\n",train_one_hot.numpy(),"\n\nand recon_y is\n",recon_y.numpy(),"\n\nand train_y_def is\n",train_one_hot_def.numpy())
            print("and score 1 is",score1.numpy(),"and score 2 is",score2.numpy(),"and train_x is\n",train_x.numpy(),"\n\nand recon_x is\n",recon_x.numpy(),"\n\nand train_x_def is\n",train_x_def.numpy(),"and train_y is\n",train_one_hot.numpy(),"\n\nand recon_y is\n",recon_y.numpy(),"\n\nand train_y_def is\n",train_one_hot_def.numpy(),file=data_print)
        # update global weights
        w_glob = FedAvg(w_locals)
        # copy weight to net_glob
        if args.local_bs!=1 and args.local_bs!=10 and args.local_bs!=5: net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        
        if args.eval==0: continue
        
        #eval  
        #net_glob=torchvision.models.resnet18()
        net_glob.to(device)
        test_label=test_data[:args.test_num,-1]
        test_x=test_data[:args.test_num,:-1]
        maxx=np.max(test_x,axis=0)
        minn=np.min(test_x,axis=0)
        test_x=(test_x-minn)/(maxx-minn+1e-9)
        log_probs = net_glob(torch.from_numpy(test_x).type(torch.FloatTensor).to(device)).cpu().detach().numpy()
        result=np.argmax(log_probs,axis=1)
        acc=np.mean(test_label==result)
        print('Round {:3d}, Average loss {:.3f}, Acc {:.3f}'.format(cur_i, loss_avg, acc),file=data_print)
        print('Round {:3d}, Average loss {:.3f}, Acc {:.3f}'.format(cur_i, loss_avg, acc))
        train_loss_plot.append(loss_avg)
        test_acc_plot.append(acc)
        
        auc_print = open("AucRecord.txt".format(args.dataset),'a+',encoding="utf-8")
        auc_prob=[1.]*len(log_probs)-log_probs[:,0]
        auc_label=copy.deepcopy(test_label)
        auc_label[auc_label!=0]=1
        auc_hat=copy.deepcopy(result)
        auc_hat[auc_hat!=0]=1
        
        TP,FP,TN,FN=compute_pos_neg(auc_label, auc_hat)
        TPR,FPR,F1,precision,recall=metrics(TP,FP,TN,FN)
        
        auc=roc_auc_score(auc_label, auc_prob)
        #if args.defense!=1:
        #    print("dataset_{}_defense_{}:\nTPR:{},FPR:{},F1:{},AUC:{}".format(args.dataset,args.defense,TPR,FPR,F1,auc),file=auc_print)
        #else: print("dataset_{}_defense_{}_alpha_{}:\nTPR:{},FPR:{},F1:{},AUC:{}".format(
        #    args.dataset,args.defense,alpha,TPR,FPR,F1,auc),file=auc_print)
        
    
    time_elapsed = time.time() - start
    print(time_elapsed,"s")
    print(time_elapsed,"s",file=data_print)
    
    alpha=int(args.alpha*100)
    
    if args.save_change==1:
        if args.attack!=0:
            if args.defense==1:
                np.save('./save/score_change_attack_{}_defense_{}_alpha_{}_dataset_{}.npy'.format(
                    args.attack,args.defense,alpha,args.dataset),arr=score1_plot)
                np.save('./save/acc_change_attack_{}_defense_{}_alpha_{}_dataset_{}.npy'.format(
                    args.attack,args.defense,alpha,args.dataset),arr=score2_plot)
            else:
                np.save('./save/score_change_attack_{}_defense_{}_dataset_{}.npy'.format(
                    args.attack,args.defense,args.dataset),arr=score1_plot)
                np.save('./save/acc_change_attack_{}_defense_{}_dataset_{}.npy'.format(
                    args.attack,args.defense,args.dataset),arr=score2_plot)
                
    if args.save_ablation==1:
        if args.attack!=0:
            np.save('./save_ablation/score_load_{}_attack_{}_deflr_{}_defepochs_{}_dataset_{}.npy'.format(args.load_model,
                args.attack,int(100*(args.defense_lr)),args.defense_epochs,args.dataset),arr=score1_plot)
            np.save('./save_ablation/acc_load_{}_attack_{}_deflr_{}_defepochs_{}_dataset_{}.npy'.format(args.load_model,
                args.attack,int(100*(args.defense_lr)),args.defense_epochs,args.dataset),arr=score2_plot)

    if args.save_model==1: 
        if args.local_bs!=1:
            if args.defense!=1:
                with open('./save_model/{}_{}_{}_model.pkl'.format(
                    args.dataset,args.model,args.defense), 'wb') as f: pickle.dump(net_glob, f)
            else:
                with open('./save_model/{}_{}_{}_alpha_{}_model.pkl'.format(
                    args.dataset,args.model,args.defense,alpha), 'wb') as f: pickle.dump(net_glob, f)
                    
    if args.save_plot==1: 
        if args.attack!=0:
            if args.defense==1:
                np.save('./save/score1_load_{}_attack_{}_defense_{}_alpha_{}_dataset_{}.npy'.format(
                    args.load_model,args.attack,args.defense,alpha,args.dataset),arr=score1_plot)
                np.save('./save/score2_load_{}_attack_{}_defense_{}_alpha_{}_dataset_{}.npy'.format(
                    args.load_model,args.attack,args.defense,alpha,args.dataset),arr=score2_plot)
            else:
                np.save('./save/score1_load_{}_attack_{}_defense_{}_dataset_{}.npy'.format(
                    args.load_model,args.attack,args.defense,args.dataset),arr=score1_plot)
                np.save('./save/score2_load_{}_attack_{}_defense_{}_dataset_{}.npy'.format(
                    args.load_model,args.attack,args.defense,args.dataset),arr=score2_plot)
        else:
            if args.defense==1:
                np.save('./save/train_loss_defense_{}_alpha_{}_dataset_{}.npy'.format(
                    args.defense,alpha,args.dataset),arr=train_loss_plot)
                np.save('./save/test_acc_defense_{}_alpha_{}_dataset_{}.npy'.format(
                    args.defense,alpha,args.dataset),arr=test_acc_plot)
            else:
                np.save('./save/train_loss_defense_{}_dataset_{}.npy'.format(args.defense,args.dataset),arr=train_loss_plot)
                np.save('./save/test_acc_defense_{}_dataset_{}.npy'.format(args.defense,args.dataset),arr=test_acc_plot)