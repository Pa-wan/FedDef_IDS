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

def transform(x,y,num_class):
    x=torch.clamp(x, min=0,max=1)
    index=torch.max(y,1)[1]
    y = F.one_hot(index, num_classes=num_class).int()
    
    return x,y

def defense(ori_x,ori_y,model,defense_epochs,alpha):
    x=torch.rand(ori_x.size()).to(device).requires_grad_(True)
    y=torch.rand(ori_y.size()).to(device).requires_grad_(True)
    #alpha=torch.ones((1)).to(device).requires_grad_(True)
    
    #x=ori_x.cpu().numpy()#+np.random.laplace(0.1,1e-1, size=x.shape)
    #y=ori_y.cpu().numpy()#+np.random.laplace(0.1,1e-1, size=y.shape)
    #x=torch.from_numpy(x).type(torch.FloatTensor).to(device).requires_grad_(True)
    #y=torch.from_numpy(y).type(torch.FloatTensor).to(device).requires_grad_(True)
    
    lr=args.defense_lr

    alpha_temp=copy.deepcopy(alpha)
    
    ori_log_probs=model(ori_x)

    ori_model_loss=-ori_y*torch.log(ori_log_probs+1e-9)
    ori_model_loss=torch.sum(ori_model_loss,dim=1)
    ori_model_loss=torch.mean(ori_model_loss)
    ori_model_loss_temp=ori_model_loss.item()
    ori_grad=torch.autograd.grad(ori_model_loss, model.parameters())
    ori_grad=[i.detach() for i in ori_grad]

    optimizer = torch.optim.Adam([x,y], lr=lr)
    index1=torch.argmax(ori_y,dim=1)
    
    for i in range(defense_epochs):
        def closure():
            x.requires_grad=True
            y.requires_grad=True
            optimizer.zero_grad()
        
            log_probs=model(x)
            model_loss=-y*torch.log(log_probs+1e-9)
            model_loss=torch.sum(model_loss,dim=1)
            model_loss=torch.mean(model_loss)
            cur_grad=torch.autograd.grad(model_loss, model.parameters(),create_graph=True)

            loss=0
            flag=0
            for j in range(len(ori_grad)): 
                if torch.max(abs(cur_grad[j]))<=args.g_value:
                    flag=1
                loss+=torch.sqrt(torch.mean((ori_grad[j]-cur_grad[j])**2)+1e-9)
            dis1=F.relu(1-torch.sqrt(torch.mean((x-ori_x)**2,dim=1)+1e-9))
            #dis1=F.relu(1-torch.sqrt(torch.mean((torch.clamp(temp_x, min=0,max=1)-temp_ori_x)**2,dim=1)+1e-9))
            dis2=abs(torch.min(y,dim=1).values-y[range(len(index1)),index1])
            temp=dis1+dis2
            
            #loss/=len(ori_grad)
            
            temp=torch.mean(temp)
            loss_all=alpha*loss+temp
            loss_all.backward()
            return loss,loss_all,flag
        diff,loss,flag=closure()
        
        if flag==1: 
            #print("Defense cur i is ",i,"and diff is",diff.item(),"and loss is",loss.item())
            return x.detach_(),y.detach_(),diff
        
        optimizer.step()

        x.detach_()
        y.detach_()
        
        #print("Defense cur i is ",i,"and loss_all is",loss.item(),"and x is",x[0],"and ori x is",ori_x[0])
    #x,y=transform(x,y,ori_y.shape[1])
    return x,y,diff
    
def defense_cv(ori_x,ori_y,net):
    gt_data=copy.deepcopy(ori_x)
    gt_onehot_label=copy.deepcopy(ori_y)

    gt_data.requires_grad = True

    # compute ||dr/dX||/||r|| 
    out= net(gt_data)
    feature_fc1_graph=net.cv(gt_data)
    #feature_fc1_graph=gt_data+1e-9
    deviation_f1_target = torch.zeros_like(feature_fc1_graph)
    deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)
    for f in range(deviation_f1_x_norm.size(1)):
        deviation_f1_target[:,f] = 1
        feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)
        deviation_f1_x = gt_data.grad.data
        deviation_f1_x_norm[:,f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1)/(feature_fc1_graph.data[:,f]+1e-9)
        net.zero_grad()
        gt_data.grad.data.zero_()
        deviation_f1_target[:,f] = 0

    # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
    deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
    #print(deviation_f1_x_norm_sum)
    thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), 80)
    #print(thresh)
    mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
    #print(sum(mask))
    return mask

def label_to_onehot(target, num_classes):
    '''Returns one-hot embeddings of scaler labels'''
    target=torch.from_numpy(target).type(torch.int64) 
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(
        0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def vec_mul_ten(vec, tensor):
    size = list(tensor.shape)
    size[0] = -1
    size_rs = [1 for i in range(len(size))]
    size_rs[0] = -1
    vec = vec.reshape(size_rs).expand(size)
    res = vec * tensor
    return res

def mixup_criterion(args,ys, lam_batch, num_class):
    '''Returns mixup loss'''
    ys_onehot = [label_to_onehot(y, num_classes=num_class) for y in ys]
    mixy = vec_mul_ten(lam_batch[:, 0], ys_onehot[0])
    for i in range(1, args.klam):
        mixy += vec_mul_ten(lam_batch[:, i], ys_onehot[i])
    return mixy

def mixup_data(args,x, y,num_class,use_cuda=True):
    '''Returns mixed inputs, lists of targets, and lambdas'''
    lams = np.random.normal(0, 1, size=(x.shape[0], args.klam))
    for i in range(x.shape[0]):
        lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))
        if args.klam > 1:
            while lams[i].max() > args.upper:     # upper bounds a single lambda
                lams[i] = np.random.normal(0, 1, size=(1, args.klam))
                lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))

    lams = torch.from_numpy(lams).float()

    mixed_x = vec_mul_ten(lams[:, 0], x)
    ys = [y]

    for i in range(1, args.klam):
        batch_size = x.shape[0]
        index = torch.randperm(batch_size)
        mixed_x += vec_mul_ten(lams[:, i], x[index, :])
        ys.append(y[index])

    sign = torch.randint(2, size=list(x.shape)) * 2.0 - 1
    mixed_x *= sign.float()
        
    return mixed_x, ys, lams

def generate_sample(args,dataset,num_class):
    np.random.shuffle(dataset)
    inputs=dataset[:1000,:-1]
    targets=dataset[:1000,-1]
        
    #print(inputs,targets)
        
    mix_inputs, mix_targets, lams = mixup_data(args,inputs, targets,num_class)
        
    return (mix_inputs, mix_targets, lams)

def defense_instahide(args,dataset,num_class):
    # You can add your own dataloader and preprocessor here.
    
    #print(dataset[:],dataset[:].shape)
    
    mix_inputs_all, mix_targets_all, lams = generate_sample(args,copy.deepcopy(dataset),num_class)
    
    y=mixup_criterion(args,mix_targets_all, lams, num_class)
    
    #print(mix_inputs_all,y.shape)
    
    return mix_inputs_all.float()[:args.local_bs].to(device),y[:args.local_bs].to(device)