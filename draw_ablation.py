import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import copy
from parse import parse_arg
import numpy as np
from torchvision import datasets, transforms
import torch
import pickle
from model.Update import LocalUpdate
from model.Nets import MLP, CNNMnist, CNNCifar, DNN, LeNet
from model.Fed import FedAvg
from model.test import test_img
from reconstruct import reconstruct

print(torch.cuda.is_available())


def draw_attack(args):
    plt.figure()

    score1_0 = np.load('./save/score1_load_{}_attack_{}_defense_0_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_0 = np.load('./save/score2_load_{}_attack_{}_defense_0_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    temp = len(score1_0) // 10

    x_list = [0]
    y_list=[0]

    for i in range(temp,args.epochs+temp,temp):
        x_list.append(i)
        y_list.append(i-1)

    x_major_locator = MultipleLocator(temp)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    score1_1_100 = np.load('./save/score1_load_{}_attack_{}_defense_1_alpha_100_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score1_1_50 = np.load('./save/score1_load_{}_attack_{}_defense_1_alpha_50_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score1_1_25 = np.load('./save/score1_load_{}_attack_{}_defense_1_alpha_25_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    score2_1_100 = np.load('./save/score2_load_{}_attack_{}_defense_1_alpha_100_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_1_50 = np.load('./save/score2_load_{}_attack_{}_defense_1_alpha_50_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_1_25 = np.load('./save/score2_load_{}_attack_{}_defense_1_alpha_25_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    score1_2 = np.load('./save/score1_load_{}_attack_{}_defense_2_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_2 = np.load('./save/score2_load_{}_attack_{}_defense_2_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    score1_3 = np.load('./save/score1_load_{}_attack_{}_defense_3_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_3 = np.load('./save/score2_load_{}_attack_{}_defense_3_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    score1_4 = np.load('./save/score1_load_{}_attack_{}_defense_4_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_4 = np.load('./save/score2_load_{}_attack_{}_defense_4_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    score1_5 = np.load('./save/score1_load_{}_attack_{}_defense_5_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_5 = np.load('./save/score2_load_{}_attack_{}_defense_5_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    data_print = open("./save/Label_Accuracy_load_{}_attack_{}_dataset_{}.txt".format(
        args.load_model, args.attack, args.dataset), 'w', encoding="utf-8")

    print("defense 0:", np.sum(score2_0) / len(score2_0), "\ndefense 1 alpha 1:", np.sum(score2_1_100) / len(score2_0),
          "\ndefense 1 alpha 0.5:",
          np.sum(score2_1_50) / len(score2_0), "\ndefense 1 alpha 0.25:", np.sum(score2_1_25) / len(score2_0),
          "\ndefense 2:",
          np.sum(score2_2) / len(score2_0), "\ndefense 3:", np.sum(score2_3) / len(score2_0), "\ndefense 4:",
          np.sum(score2_4) / len(score2_0), "\ndefense 5:", np.sum(score2_5) / len(score2_0), file=data_print)

    score1_0 = np.squeeze(score1_0)
    score1_1_100 = np.squeeze(score1_1_100)
    score1_1_50 = np.squeeze(score1_1_50)
    score1_1_25 = np.squeeze(score1_1_25)
    score1_2 = np.squeeze(score1_2)
    score1_3 = np.squeeze(score1_3)
    score1_4 = np.squeeze(score1_4)
    score1_5 = np.squeeze(score1_5)

    plt.plot(x_list, score1_0[y_list], marker='^',markersize='10',alpha=1,label='No Defense')

    plt.plot(x_list, score1_1_100[y_list], marker='s',markersize='10',alpha=1,label='Our Defense alpha 1')
    plt.plot(x_list, score1_1_50[y_list], marker='s',markersize='10',alpha=1,label='Our Defense alpha 0.5')
    plt.plot(x_list, score1_1_25[y_list], marker='s',markersize='10',alpha=1,label='Our Defense alpha 0.25')

    plt.plot(x_list, score1_2[y_list], marker='p',markersize='10',alpha=1,label='Soteria')
    plt.plot(x_list, score1_3[y_list], marker='x',markersize='10',alpha=1,label='Gradient Pruning')
    plt.plot(x_list, score1_4[y_list], marker='o',markersize='10',alpha=1,label='Differential Privacy')
    plt.plot(x_list, score1_5[y_list], marker='D',markersize='10',alpha=1,label='Instahide')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Reconstruction Feature Score')
    plt.grid()

    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width*1, box.height * 0.80])
    #plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=8)
    plt.savefig('./save/score1_load_{}_attack_{}_dataset_{}.pdf'.format(args.load_model, args.attack, args.dataset))
    print("Done")
    
def draw_attack_bar(args):
    plt.rc('axes', axisbelow=True)
    fig = plt.figure(figsize=(6 * 1.8, 6 * 0.718), constrained_layout=True)

    axs = fig.subplots(1, 1)
    score_all=[]
    for i in range(4):
        if i ==0: 
            args.load_model=0
            args.attack=1
        if i ==1: 
            args.load_model=0
            args.attack=2
        if i ==2: 
            args.load_model=1
            args.attack=1
        if i ==3: 
            args.load_model=1
            args.attack=2
        temp=[]
        score1_0 = np.load('./save/score1_load_{}_attack_{}_defense_0_dataset_{}.npy'.format(
            args.load_model, args.attack, args.dataset))
        score1_1_100 = np.load('./save/score1_load_{}_attack_{}_defense_1_alpha_100_dataset_{}.npy'.format(
            args.load_model, args.attack, args.dataset))
        score1_1_50 = np.load('./save/score1_load_{}_attack_{}_defense_1_alpha_50_dataset_{}.npy'.format(
            args.load_model, args.attack, args.dataset))
        score1_1_25 = np.load('./save/score1_load_{}_attack_{}_defense_1_alpha_25_dataset_{}.npy'.format(
            args.load_model, args.attack, args.dataset))
        score1_2 = np.load('./save/score1_load_{}_attack_{}_defense_2_dataset_{}.npy'.format(
            args.load_model, args.attack, args.dataset))
        score1_3 = np.load('./save/score1_load_{}_attack_{}_defense_3_dataset_{}.npy'.format(
            args.load_model, args.attack, args.dataset))
        score1_4 = np.load('./save/score1_load_{}_attack_{}_defense_4_dataset_{}.npy'.format(
            args.load_model, args.attack, args.dataset))
        score1_5 = np.load('./save/score1_load_{}_attack_{}_defense_5_dataset_{}.npy'.format(
            args.load_model, args.attack, args.dataset))
    
        temp.append(np.squeeze(score1_0))
        temp.append(np.squeeze(score1_1_100))
        temp.append(np.squeeze(score1_1_50))
        temp.append(np.squeeze(score1_1_25))
        temp.append(np.squeeze(score1_2))
        temp.append(np.squeeze(score1_3))
        temp.append(np.squeeze(score1_4))
        temp.append(np.squeeze(score1_5))
        score_all.append(temp)
        
    
    label = ["No Defense", "Our Defense alpha 1", "Our Defense alpha 0.5","Our Defense alpha 0.25",
             "Soteria","Gradient Pruning","Differential Privacy","Instahide"]
    name = ["Early+Optimization Attack",
        "Early+Extraction Attack",
        "Late+Optimization Attack",
        "Late+Extraction Attack"]
    rate = ["(1:1)", "(1:2)", "(1:4)", "(1:8)"]

    x = np.array([1 * i for i in range(len(name))])
    width = 0.08

    data_0 = [np.mean(score_all[j][0]) for j in range(len(name))]
    data_1_100 = [np.mean(score_all[j][1]) for j in range(len(name))]
    data_1_50 = [np.mean(score_all[j][2]) for j in range(len(name))]
    data_1_25 = [np.mean(score_all[j][3]) for j in range(len(name))]
    data_2 = [np.mean(score_all[j][4]) for j in range(len(name))]
    data_3 = [np.mean(score_all[j][5]) for j in range(len(name))]
    data_4 = [np.mean(score_all[j][6]) for j in range(len(name))]
    data_5 = [np.mean(score_all[j][7]) for j in range(len(name))]
    
    min_data_0 = [data_0[j]-np.min(score_all[j][0]) for j in range(len(name))]
    min_data_1_100 = [data_1_100[j]-np.min(score_all[j][1]) for j in range(len(name))]
    min_data_1_50 = [data_1_50[j]-np.min(score_all[j][2]) for j in range(len(name))]
    min_data_1_25 = [data_1_25[j]-np.min(score_all[j][3]) for j in range(len(name))]
    min_data_2 = [data_2[j]-np.min(score_all[j][4]) for j in range(len(name))]
    min_data_3 = [data_3[j]-np.min(score_all[j][5]) for j in range(len(name))]
    min_data_4 = [data_4[j]-np.min(score_all[j][6]) for j in range(len(name))]
    min_data_5 = [data_5[j]-np.min(score_all[j][7]) for j in range(len(name))]
    
    max_data_0 = [np.max(score_all[j][0])-data_0[j] for j in range(len(name))]
    max_data_1_100 = [np.max(score_all[j][1])-data_1_100[j] for j in range(len(name))]
    max_data_1_50 = [np.max(score_all[j][2])-data_1_50[j] for j in range(len(name))]
    max_data_1_25 = [np.max(score_all[j][3])-data_1_25[j] for j in range(len(name))]
    max_data_2 = [np.max(score_all[j][4])-data_2[j] for j in range(len(name))]
    max_data_3 = [np.max(score_all[j][5])-data_3[j] for j in range(len(name))]
    max_data_4 = [np.max(score_all[j][6])-data_4[j] for j in range(len(name))]
    max_data_5 = [np.max(score_all[j][7])-data_5[j] for j in range(len(name))]

    err_0=[min_data_0,max_data_0]
    err_1_100=[min_data_1_100,max_data_1_100]
    err_1_50=[min_data_1_50,max_data_1_50]
    err_1_25=[min_data_1_25,max_data_1_25]
    err_2=[min_data_2,max_data_2]
    err_3=[min_data_3,max_data_3]
    err_4=[min_data_4,max_data_4]
    err_5=[min_data_5,max_data_5]
    
    axs.bar(x - 3*width * 1.1, data_0, width,  error_kw={'lw': 2, 'capsize': 6},
                 yerr=err_0, label=label[0],alpha=0.5, lw=2)
    axs.bar(x - 2*width * 1.1, data_1_100, width,  error_kw={'lw': 2, 'capsize': 6},
                 yerr=err_1_100, label=label[1], alpha=0.5, lw=2)
    axs.bar(x - width * 1.1, data_1_50, width,  error_kw={'lw': 2, 'capsize': 6},
                 yerr=err_1_50, label=label[2], alpha=0.5, lw=2)
    axs.bar(x, data_1_25, width,  error_kw={'lw': 2, 'capsize': 6},
                 yerr=err_1_25, label=label[3], alpha=0.5, lw=2)
    axs.bar(x + width * 1.1, data_2, width,  error_kw={'lw': 2, 'capsize': 6},
                 yerr=err_2, label=label[4], alpha=0.5, lw=2)
    axs.bar(x + 2*width * 1.1, data_3, width,  error_kw={'lw': 2, 'capsize': 6},
                 yerr=err_3, label=label[5], alpha=0.5, lw=2)
    axs.bar(x + 3*width * 1.1, data_4, width,  error_kw={'lw': 2, 'capsize': 6},
                 yerr=err_4, label=label[6], alpha=0.5, lw=2)
    axs.bar(x + 4*width * 1.1, data_5, width,  error_kw={'lw': 2, 'capsize': 6},
                 yerr=err_5, label=label[7], alpha=0.5, lw=2)

    box = axs.get_position()
    axs.set_position([box.x0, box.y0, box.width*1, box.height * 0.80])
    plt.legend(loc='center left', bbox_to_anchor=(0.0, 1.25), ncol=4)

    axs.set_xticks(x)
    axs.set_ylabel('Score', fontsize=10)

    plt.tick_params(labelsize=10)
    axs.set_xticklabels(name)

    #y_major_locator = MultipleLocator(0.1)
    #axs.yaxis.set_major_locator(y_major_locator)

    plt.savefig('./save/score_dataset_{}.pdf'.format(args.dataset))
    print("Done")


def draw_acc(args):
    plt.figure()

    train_loss_0 = np.load('./save/train_loss_defense_{}_dataset_{}.npy'.format(0, args.dataset))
    test_acc_0 = np.load('./save/test_acc_defense_{}_dataset_{}.npy'.format(0, args.dataset))

    temp = len(train_loss_0) // 10

    x_list = [0]
    y_list = [0]

    for i in range(temp, args.epochs + temp, temp):
        x_list.append(i)
        y_list.append(i - 1)
        
    x_list = range(args.epochs)
    y_list = range(args.epochs)

    x_major_locator = MultipleLocator(temp)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    train_loss_1_100 = np.load('./save/train_loss_defense_{}_alpha_{}_dataset_{}.npy'.format(1, 100, args.dataset))
    train_loss_1_50 = np.load('./save/train_loss_defense_{}_alpha_{}_dataset_{}.npy'.format(1, 50, args.dataset))
    train_loss_1_25 = np.load('./save/train_loss_defense_{}_alpha_{}_dataset_{}.npy'.format(1, 25, args.dataset))
    test_acc_1_100 = np.load('./save/test_acc_defense_{}_alpha_{}_dataset_{}.npy'.format(1, 100, args.dataset))
    test_acc_1_50 = np.load('./save/test_acc_defense_{}_alpha_{}_dataset_{}.npy'.format(1, 50, args.dataset))
    test_acc_1_25 = np.load('./save/test_acc_defense_{}_alpha_{}_dataset_{}.npy'.format(1, 25, args.dataset))

    train_loss_2 = np.load('./save/train_loss_defense_{}_dataset_{}.npy'.format(2, args.dataset))
    test_acc_2 = np.load('./save/test_acc_defense_{}_dataset_{}.npy'.format(2, args.dataset))

    train_loss_3 = np.load('./save/train_loss_defense_{}_dataset_{}.npy'.format(3, args.dataset))
    test_acc_3 = np.load('./save/test_acc_defense_{}_dataset_{}.npy'.format(3, args.dataset))

    train_loss_4 = np.load('./save/train_loss_defense_{}_dataset_{}.npy'.format(4, args.dataset))
    test_acc_4 = np.load('./save/test_acc_defense_{}_dataset_{}.npy'.format(4, args.dataset))

    train_loss_5 = np.load('./save/train_loss_defense_{}_dataset_{}.npy'.format(5, args.dataset))
    test_acc_5 = np.load('./save/test_acc_defense_{}_dataset_{}.npy'.format(5, args.dataset))

    data_print = open("./save/Model_Accuracy_attack_{}_dataset_{}.txt".format(args.attack, args.dataset), 'w',
                      encoding="utf-8")

    print("defense 0:", test_acc_0[-5:], "\ndefense 1 alpha 1:", test_acc_1_100[-5:], "\ndefense 1 alpha 0.5:",
          test_acc_1_50[-5:], "\ndefense 1 alpha 0.25:", test_acc_1_25[-5:], "\ndefense 2:",
          test_acc_2[-5:], "\ndefense 3:", test_acc_3[-5:], "\ndefense 4:",
          test_acc_4[-5:], "\ndefense 5:", test_acc_5[-5:], file=data_print)

    plt.plot(x_list, train_loss_0[y_list],label='No Defense')

    plt.plot(x_list, train_loss_1_100[y_list], label='Our Defense alpha 1')
    plt.plot(x_list, train_loss_1_50[y_list], label='Our Defense alpha 0.5')
    plt.plot(x_list, train_loss_1_25[y_list], label='Our Defense alpha 0.25')

    plt.plot(x_list, train_loss_2[y_list], label='Soteria')
    plt.plot(x_list, train_loss_3[y_list], label='Gradient Pruning')
    plt.plot(x_list, train_loss_4[y_list], label='Differential Privacy')
    plt.plot(x_list, train_loss_5[y_list], label='Instahide')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid()

    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width, box.height * 0.80])
    #plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=2)

    plt.savefig('./save/train_loss_{}_ori.pdf'.format(args.dataset))

    plt.figure()
    x_major_locator = MultipleLocator(temp)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.plot(x_list, test_acc_0[y_list], label='No Defense')

    plt.plot(x_list, test_acc_1_100[y_list], label='Our Defense alpha 1')
    plt.plot(x_list, test_acc_1_50[y_list], label='Our Defense alpha 0.5')
    plt.plot(x_list, test_acc_1_25[y_list], label='Our Defense alpha 0.25')

    plt.plot(x_list, test_acc_2[y_list], label='Soteria')
    plt.plot(x_list, test_acc_3[y_list], label='Gradient Pruning')
    plt.plot(x_list, test_acc_4[y_list], label='Differential Privacy')
    plt.plot(x_list, test_acc_5[y_list], label='Instahide')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('Test Accuracy')
    plt.grid()

    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width, box.height * 0.80])
    #plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=2)

    plt.savefig('./save/test_acc_{}_ori.pdf'.format(args.dataset))

    print("Done")
    
def draw_gan(args):
    plt.figure()

    acc_plot_0 = np.load('./gan_dataset/save/acc_{}_0.npy'.format(args.dataset))
    rmse_plot_0 = np.load('./gan_dataset/save/rmse_{}_0.npy'.format(args.dataset))
    
    acc_plot_1_100 = np.load('./gan_dataset/save/acc_{}_1_alpha_100.npy'.format(args.dataset))
    rmse_plot_1_100 = np.load('./gan_dataset/save/rmse_{}_1_alpha_100.npy'.format(args.dataset))
    acc_plot_1_50 = np.load('./gan_dataset/save/acc_{}_1_alpha_50.npy'.format(args.dataset))
    rmse_plot_1_50 = np.load('./gan_dataset/save/rmse_{}_1_alpha_50.npy'.format(args.dataset))
    acc_plot_1_25 = np.load('./gan_dataset/save/acc_{}_1_alpha_25.npy'.format(args.dataset))
    rmse_plot_1_25 = np.load('./gan_dataset/save/rmse_{}_1_alpha_25.npy'.format(args.dataset))
    
    acc_plot_2 = np.load('./gan_dataset/save/acc_{}_2.npy'.format(args.dataset))
    rmse_plot_2 = np.load('./gan_dataset/save/rmse_{}_2.npy'.format(args.dataset))
    
    acc_plot_3 = np.load('./gan_dataset/save/acc_{}_3.npy'.format(args.dataset))
    rmse_plot_3 = np.load('./gan_dataset/save/rmse_{}_3.npy'.format(args.dataset))
    
    acc_plot_4 = np.load('./gan_dataset/save/acc_{}_4.npy'.format(args.dataset))
    rmse_plot_4 = np.load('./gan_dataset/save/rmse_{}_4.npy'.format(args.dataset))
    
    acc_plot_5 = np.load('./gan_dataset/save/acc_{}_5.npy'.format(args.dataset))
    rmse_plot_5 = np.load('./gan_dataset/save/rmse_{}_5.npy'.format(args.dataset))

    temp = len(acc_plot_0) // 10

    x_list = range(args.n_epochs)
    y_list = range(args.n_epochs)

    x_major_locator = MultipleLocator(temp)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.plot(x_list, acc_plot_0[y_list],label='No Defense')
    plt.plot(x_list, acc_plot_1_100[y_list],label='Our Defense alpha 1')
    plt.plot(x_list, acc_plot_1_50[y_list],label='Our Defense alpha 0.5')
    plt.plot(x_list, acc_plot_1_25[y_list],label='Our Defense alpha 0.25')
    plt.plot(x_list, acc_plot_2[y_list],label='Soteria')
    plt.plot(x_list, acc_plot_3[y_list],label='Gradient Pruning')
    plt.plot(x_list, acc_plot_4[y_list],label='Differential Privacy')
    plt.plot(x_list, acc_plot_5[y_list],label='Instahide')
    
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('DNN Accuracy For Adversarial Examples')
    plt.grid()

    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width, box.height * 0.80])
    #plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=2)

    plt.savefig('./gan_dataset/save/acc_{}.pdf'.format(args.dataset))

    plt.figure()
    x_major_locator = MultipleLocator(temp)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.plot(x_list, rmse_plot_0[y_list],label='No Defense')
    plt.plot(x_list, rmse_plot_1_100[y_list],label='Our Defense alpha 1')
    plt.plot(x_list, rmse_plot_1_50[y_list],label='Our Defense alpha 0.5')
    plt.plot(x_list, rmse_plot_1_25[y_list],label='Our Defense alpha 0.25')
    plt.plot(x_list, rmse_plot_2[y_list],label='Soteria')
    plt.plot(x_list, rmse_plot_3[y_list],label='Gradient Pruning')
    plt.plot(x_list, rmse_plot_4[y_list],label='Differential Privacy')
    plt.plot(x_list, rmse_plot_5[y_list],label='Instahide')
    
    temp=0
    
    if args.dataset=='kdd': temp=0.25
    elif args.dataset=='mirai': temp=0.6
    elif args.dataset=='cic2017': temp=0.08
    elif args.dataset=='unsw': temp=0.21
        
    
    plt.plot(x_list,[temp]*len(x_list),c='black',linewidth=2,label="Threshold")
    
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Kitsune RMSE For Adversarial Examples')
    plt.grid()

    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width, box.height * 0.80])
    #plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=2)

    plt.savefig('./gan_dataset/save/rmse_{}.pdf'.format(args.dataset))

    print("Done")

def draw_change(args):
    plt.figure()

    score1_0 = np.load('./save/score_change_load_{}_attack_{}_defense_0_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_0 = np.load('./save/acc_change_load_{}_attack_{}_defense_0_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    temp = len(score1_0) // 10

    x_list = [0]
    y_list=[0]

    for i in range(temp,args.epochs+temp,temp):
        x_list.append(i)
        y_list.append(i-1)

    x_major_locator = MultipleLocator(temp)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    score1_1_100 = np.load('./save/score_change_load_{}_attack_{}_defense_1_alpha_100_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score1_1_50 = np.load('./save/score_change_load_{}_attack_{}_defense_1_alpha_50_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score1_1_25 = np.load('./save/score_change_load_{}_attack_{}_defense_1_alpha_25_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    score2_1_100 = np.load('./save/acc_change_load_{}_attack_{}_defense_1_alpha_100_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_1_50 = np.load('./save/acc_change_load_{}_attack_{}_defense_1_alpha_50_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_1_25 = np.load('./save/acc_change_load_{}_attack_{}_defense_1_alpha_25_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    score1_2 = np.load('./save/score_change_load_{}_attack_{}_defense_2_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_2 = np.load('./save/acc_change_load_{}_attack_{}_defense_2_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    score1_3 = np.load('./save/score_change_load_{}_attack_{}_defense_3_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_3 = np.load('./save/acc_change_load_{}_attack_{}_defense_3_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    score1_4 = np.load('./save/score_change_load_{}_attack_{}_defense_4_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_4 = np.load('./save/acc_change_load_{}_attack_{}_defense_4_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    score1_5 = np.load('./save/score_change_load_{}_attack_{}_defense_5_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))
    score2_5 = np.load('./save/acc_change_load_{}_attack_{}_defense_5_dataset_{}.npy'.format(
        args.load_model, args.attack, args.dataset))

    score1_0 = np.squeeze(score1_0)
    score1_1_100 = np.squeeze(score1_1_100)
    score1_1_50 = np.squeeze(score1_1_50)
    score1_1_25 = np.squeeze(score1_1_25)
    score1_2 = np.squeeze(score1_2)
    score1_3 = np.squeeze(score1_3)
    score1_4 = np.squeeze(score1_4)
    score1_5 = np.squeeze(score1_5)
    
    score2_0 = np.squeeze(score2_0)
    score2_1_100 = np.squeeze(score2_1_100)
    score2_1_50 = np.squeeze(score2_1_50)
    score2_1_25 = np.squeeze(score2_1_25)
    score2_2 = np.squeeze(score2_2)
    score2_3 = np.squeeze(score2_3)
    score2_4 = np.squeeze(score2_4)
    score2_5 = np.squeeze(score2_5)

    plt.plot(x_list, score1_0[y_list], marker='^',markersize='10',alpha=1,label='No Defense')

    plt.plot(x_list, score1_1_100[y_list], marker='s',markersize='10',alpha=1,label='Our Defense alpha 1')
    plt.plot(x_list, score1_1_50[y_list], marker='s',markersize='10',alpha=1,label='Our Defense alpha 0.5')
    plt.plot(x_list, score1_1_25[y_list], marker='s',markersize='10',alpha=1,label='Our Defense alpha 0.25')

    plt.plot(x_list, score1_2[y_list], marker='p',markersize='10',alpha=1,label='Soteria')
    plt.plot(x_list, score1_3[y_list], marker='x',markersize='10',alpha=1,label='Gradient Pruning')
    plt.plot(x_list, score1_4[y_list], marker='o',markersize='10',alpha=1,label='Differential Privacy')
    plt.plot(x_list, score1_5[y_list], marker='D',markersize='10',alpha=1,label='Instahide')
    plt.xlabel('Training Epoch')
    plt.ylabel('Score')
    plt.title('Reconstruction Feature Score')
    plt.grid()

    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width*1, box.height * 0.80])
    #plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=8)
    plt.savefig('./save/score_change_attack_{}_dataset_{}.pdf'.format(args.attack, args.dataset))
    
    plt.figure()
    x_major_locator = MultipleLocator(temp)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.plot(x_list, score2_0[y_list], marker='^',markersize='10',alpha=1,label='No Defense')

    plt.plot(x_list, score2_1_100[y_list], marker='s',markersize='10',alpha=1,label='Our Defense alpha 1')
    plt.plot(x_list, score2_1_50[y_list], marker='s',markersize='10',alpha=1,label='Our Defense alpha 0.5')
    plt.plot(x_list, score2_1_25[y_list], marker='s',markersize='10',alpha=1,label='Our Defense alpha 0.25')

    plt.plot(x_list, score2_2[y_list], marker='p',markersize='10',alpha=1,label='Soteria')
    plt.plot(x_list, score2_3[y_list], marker='x',markersize='10',alpha=1,label='Gradient Pruning')
    plt.plot(x_list, score2_4[y_list], marker='o',markersize='10',alpha=1,label='Differential Privacy')
    plt.plot(x_list, score2_5[y_list], marker='D',markersize='10',alpha=1,label='Instahide')
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy')
    plt.title('Reconstruction Label Accuracy')
    plt.grid()

    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width*1, box.height * 0.80])
    #plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=8)
    plt.savefig('./save/acc_change_attack_{}_dataset_{}.pdf'.format(args.attack, args.dataset))
    
    print("Done")
    
def draw_fl(args):
    plt.rc('axes', axisbelow=True)
    fig = plt.figure(figsize=(6 * 1.8, 6 * 0.718))

    axs = fig.subplots(1, 1)
    
    name = ["FL","User-1","User-2","User-3","User-4","User-5","User-6","User-7","User-8","User-9","User-10",]

    x = np.array([i for i in range(len(name))])
    
    if args.dataset=='kdd': data=[0.937,0.309,0.344,0.319,0.544,0.670,0.200,0.597,0.748,0.400,0.803]
    elif args.dataset=='unsw': data=[0.663,0.617,0.522,0.364,0.623,0.507,0.588,0.593,0.413,0.405,0.511]
    
    #plt.plot(x,data)

    for i in range(len(x)): axs.bar(x[i], data[i],alpha=0.5)

    axs.set_xticks(x)
    axs.set_ylabel('Accuracy', fontsize=10)

    plt.tick_params(labelsize=10)
    axs.set_xticklabels(name)

    plt.savefig('./save/acc_fl_dataset_{}.pdf'.format(args.dataset))
    print("Done")
    
    
args = parse_arg()
#draw_gan(args)
#draw_attack_bar(args)
draw_fl(args)
#if args.attack == 0:
#    draw_acc(args)
#else:
#    draw_attack(args)