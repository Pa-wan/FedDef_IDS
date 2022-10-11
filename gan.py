import os
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from math import ceil
import pickle
import warnings
import copy
from parse import parse_arg
import KitNET
warnings.filterwarnings("ignore")

opt = parse_arg()
print(opt)

cuda = True if torch.cuda.is_available() else False

print(cuda)

delta=1e-9
def norm(X):
    X_min = torch.min(X,0).values
    X_max = torch.max(X,0).values
    X = (X - X_min) / (X_max - X_min + delta)
    return (X, X_min, X_max)

def norm2(X, X_min, X_max):  # X is a numpy array
    X_mins = torch.min(X,0).values
    X_maxs = torch.max(X,0).values
    X = (X - X_min) / (X_max - X_min + delta)
    return (X, X_mins, X_maxs)

def denorm(X, X_min, X_max):
    X = X * (X_max - X_min + delta) + X_min
    return X

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1*input_size, 2*input_size),
            nn.ReLU(),
            nn.Linear(2*input_size, 3*input_size),
            nn.ReLU(),
            nn.Linear(3*input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, z):
        gen_z = self.model(z)
        #print("now is gen_z "+str(gen_z))
        return gen_z


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 2*input_size),
            nn.ReLU(),
            nn.Linear(2*input_size, 2*input_size),
            nn.ReLU(),
            nn.Linear(2*input_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, z):
        prob = self.model(z)
        #print("now is prob "+str(prob))
        return prob

def calacc(prob1,x):
    #print(prob1)
    prob1[prob1>0.5]=1
    prob1[prob1<=0.5]=0
    temp1=np.sum(prob1)
    acc1=temp1/x.shape[0]
    return temp1,acc1


def transform(x,y,num_class,maxx,minn,discrete):
    x=torch.clamp(x, min=0,max=1)
    index=torch.max(y,1)[1]
    y = F.one_hot(index, num_classes=num_class).float()
    
    if len(discrete)==0: return x.detach(),y.detach()
    
    x=x*(maxx-minn+1e-9)+minn
    
    x[:,discrete]=torch.round(x[:,discrete])
    
    x=(x-minn)/(maxx-minn+1e-9)
    
    return x.detach(),y.detach()

#load kdd.npy
args=opt
alpha=int(args.alpha*100)

if opt.use_ori==1:
    x=np.load('./data/{}.npy'.format(opt.dataset))
    result=x[:,-1]
    x=x[:,:-1]
    x = torch.from_numpy(x).type(torch.FloatTensor)
    with open("./kitsune_model/__model__{}.pkl".format(opt.dataset), 'rb') as f: net = pickle.load(f)# 读取模型信息
    #x,ben_min,ben_max=norm(x)
    x=net.norm(x)
    #x=(x-minn)/(maxx-minn+1e-9)

else:
    if opt.defense!=1:
        x=np.load('./gan_dataset/x_{}_{}_{}.npy'.format(args.dataset, args.model, args.defense))
        label=np.load('./gan_dataset/y_{}_{}_{}.npy'.format(args.dataset, args.model, args.defense))
    else:
        x=np.load('./gan_dataset/x_{}_{}_{}_alpha_{}.npy'.format(args.dataset, args.model, args.defense,alpha))
        label=np.load('./gan_dataset/y_{}_{}_{}_alpha_{}.npy'.format(args.dataset, args.model, args.defense,alpha))

    #x=x[:100]
    #label=label[:100]

    result=np.argmax(label,axis=1)
    #result[result!=0]=0
    x = torch.from_numpy(x).type(torch.FloatTensor)
    discrete=[]
    #print(x,result)

ben_x=x[result==0]
mal_x=x[result!=0]
ben_x=ben_x[:1000]
mal_x=copy.deepcopy(ben_x)

print(x,x.shape)
print(ben_x.shape,mal_x.shape)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(x.shape[1])
discriminator = Discriminator(x.shape[1],1)
#if opt.pretrain==0:
#    with open("./gan_dataset/{}_{}_dis.pkl".format(opt.dataset,opt.defense), 'rb') as f: discriminator = pickle.load(f)# 读取模型信息

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.gan_lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.gan_lr, betas=(opt.b1, opt.b2))

# ----------
#  Training
# ----------

acc_plot=[]
rmse_plot=[]
pause=0
for epoch in range(opt.n_epochs):
    # Adversarial ground truths
    benign=torch.zeros((ben_x.shape[0],1))
    malicious=torch.ones((mal_x.shape[0],1))

    # ---------------------
    #  Train Discriminator
    # ---------------------
    
    optimizer_D.zero_grad()

    # Measure discriminator's ability to classify real from generated samples
    prob=discriminator(ben_x)

    temp1,acc1=calacc(copy.deepcopy(prob.detach().numpy()),ben_x)
    temp1=ben_x.shape[0]-temp1
    acc1=1-acc1
    real_loss = torch.mean(prob)
    
    z=torch.rand(mal_x.size())
    gen_z = generator(z)
    if epoch<=opt.pretrain: 
        prob=discriminator(mal_x.detach())
    else: 
        prob=discriminator(gen_z.detach())

    temp2,acc2=calacc(copy.deepcopy(prob.detach().numpy()),mal_x)
    fake_loss=-torch.mean(prob)
    
    d_loss = (real_loss + fake_loss)/1

    d_loss.backward()
    optimizer_D.step()
    
    for p in discriminator.parameters(): p.data.clamp_(-opt.clip_value, opt.clip_value)
    
    if epoch % opt.n_critic == 0:

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_z = generator(z)
        #gen_z=torch.clamp(gen_z,min=minn,max=maxx)
        # Loss measures generator's ability to fool the discriminator
        benign=torch.zeros((mal_x.shape[0],1))
        
        g_loss = torch.mean(discriminator(gen_z))#+torch.mean((gen_z-mal_x)**2)
        #g_loss = torch.mean(torch.log(discriminator(gen_z)+1e-9))

        if  epoch > opt.pretrain:
            g_loss.backward()
            if pause==0: optimizer_G.step()

    print("[Epoch %d/%d] [D loss: %f] [G loss: %f] [ACC1 : %d/%d=%f] [ACC2 : %d/%d=%f]" % (epoch+1, opt.n_epochs, d_loss.item(), (g_loss).item(), temp1, ben_x.shape[0], acc1,temp2, mal_x.shape[0], acc2))
    
    alpha=int(opt.alpha*100)
    
    if opt.defense!=1:
        with open('./save_model/{}_{}_{}_model.pkl'.format(args.dataset,args.model,opt.defense), 'rb') as f: net = pickle.load(f)
    else:
        with open('./save_model/{}_{}_{}_alpha_{}_model.pkl'.format(args.dataset,args.model,opt.defense,alpha), 'rb') as f: net = pickle.load(f)
    
    log_probs=net(gen_z)
    #print(log_probs[:3])
    result=np.argmax(log_probs.detach().numpy(),axis=1)
    
    result[result!=0]=1
    
    print("dnn acc is",np.mean(result))
    
    acc_plot.append(np.mean(result))
    
    
    with open("./kitsune_model/__model__{}.pkl".format(opt.dataset), 'rb') as f: net = pickle.load(f)# 读取模型信息
    result=net.execute(gen_z)
    idx=result<0.14
    #print(np.sum(idx))
    result=np.mean(result)
    rmse_plot.append(result)
    print(result)
    #if result<0.3: pause=1
    #if result<0.08: break
    
alpha=int(opt.alpha*100)

if opt.save_model==1: 
    if opt.defense!=1:
        np.save('./gan_dataset/save/acc_{}_{}.npy'.format(args.dataset, args.defense),arr=acc_plot)
        np.save('./gan_dataset/save/rmse_{}_{}.npy'.format(args.dataset,args.defense),arr=rmse_plot)
    else:
        np.save('./gan_dataset/save/acc_{}_{}_alpha_{}.npy'.format(args.dataset, args.defense,alpha),arr=acc_plot)
        np.save('./gan_dataset/save/rmse_{}_{}_alpha_{}.npy'.format(args.dataset,args.defense,alpha),arr=rmse_plot)
        
print(gen_z[:,:6])
ori_z=net.unnorm(gen_z)
print("ori gen_z is ",ori_z[:,:6])
np.save(file="./gan_dataset/malicious1.npy",arr=gen_z.detach().numpy())