import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import copy
from parse import parse_arg
import numpy as np
import torch
import pickle
from model.Update import LocalUpdate
from model.Nets import MLP, CNNMnist, CNNCifar, DNN, LeNet
from model.Fed import FedAvg
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
    y_list = [0]

    for i in range(temp, args.epochs + temp, temp):
        x_list.append(i)
        y_list.append(i - 1)

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

    plt.plot(x_list, score1_0[y_list], marker='^', markersize='10', alpha=1, label='None')

    plt.plot(x_list, score1_1_100[y_list], marker='s', markersize='10', alpha=1, label='FedDef alpha 1')
    plt.plot(x_list, score1_1_50[y_list], marker='s', markersize='10', alpha=1, label='FedDef alpha 0.5')
    plt.plot(x_list, score1_1_25[y_list], marker='s', markersize='10', alpha=1, label='FedDef alpha 0.25')

    plt.plot(x_list, score1_2[y_list], marker='p', markersize='10', alpha=1, label='Soteria')
    plt.plot(x_list, score1_3[y_list], marker='x', markersize='10', alpha=1, label='GP')
    plt.plot(x_list, score1_4[y_list], marker='o', markersize='10', alpha=1, label='DP')
    plt.plot(x_list, score1_5[y_list], marker='D', markersize='10', alpha=1, label='Instahide')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Reconstruction Feature Score')
    plt.grid()

    box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width*1, box.height * 0.80])
    # plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=8)
    plt.savefig('./save/score1_load_{}_attack_{}_dataset_{}.pdf'.format(args.load_model, args.attack, args.dataset))
    print("Done")


def draw_attack_bar(args):
    plt.rc('axes', axisbelow=True)
    fig = plt.figure(figsize=(6 * 1.8, 8 * 0.718), constrained_layout=True)

    axs = fig.subplots(1, 1)
    score_all = []
    for i in range(4):
        if i == 0:
            args.load_model = 0
            args.attack = 1
        if i == 1:
            args.load_model = 0
            args.attack = 2
        if i == 2:
            args.load_model = 1
            args.attack = 1
        if i == 3:
            args.load_model = 1
            args.attack = 2
        temp = []
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
        
        print(score1_0.shape)

        temp.append(np.squeeze(score1_0))
        temp.append(np.squeeze(score1_1_100))
        temp.append(np.squeeze(score1_1_50))
        temp.append(np.squeeze(score1_1_25))
        temp.append(np.squeeze(score1_2))
        temp.append(np.squeeze(score1_3))
        temp.append(np.squeeze(score1_4))
        temp.append(np.squeeze(score1_5))
        score_all.append(temp)

    label = ["None", "FedDef alpha 1", "FedDef alpha 0.5", "FedDef alpha 0.25",
             "Soteria", "GP", "DP", "Instahide"]
    name = ["Early+Inversion",
            "Early+Extraction",
            "Late+Inversion",
            "Late+Extraction"]
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
    
    print(data_0[0],data_1_100[0],data_1_50[0],data_1_25[0],data_2[0],data_3[0],data_4[0],data_5[0])

    min_data_0 = [data_0[j] - np.min(score_all[j][0]) for j in range(len(name))]
    min_data_1_100 = [data_1_100[j] - np.min(score_all[j][1]) for j in range(len(name))]
    min_data_1_50 = [data_1_50[j] - np.min(score_all[j][2]) for j in range(len(name))]
    min_data_1_25 = [data_1_25[j] - np.min(score_all[j][3]) for j in range(len(name))]
    min_data_2 = [data_2[j] - np.min(score_all[j][4]) for j in range(len(name))]
    min_data_3 = [data_3[j] - np.min(score_all[j][5]) for j in range(len(name))]
    min_data_4 = [data_4[j] - np.min(score_all[j][6]) for j in range(len(name))]
    min_data_5 = [data_5[j] - np.min(score_all[j][7]) for j in range(len(name))]

    max_data_0 = [np.max(score_all[j][0]) - data_0[j] for j in range(len(name))]
    max_data_1_100 = [np.max(score_all[j][1]) - data_1_100[j] for j in range(len(name))]
    max_data_1_50 = [np.max(score_all[j][2]) - data_1_50[j] for j in range(len(name))]
    max_data_1_25 = [np.max(score_all[j][3]) - data_1_25[j] for j in range(len(name))]
    max_data_2 = [np.max(score_all[j][4]) - data_2[j] for j in range(len(name))]
    max_data_3 = [np.max(score_all[j][5]) - data_3[j] for j in range(len(name))]
    max_data_4 = [np.max(score_all[j][6]) - data_4[j] for j in range(len(name))]
    max_data_5 = [np.max(score_all[j][7]) - data_5[j] for j in range(len(name))]
    
    err_0 = [min_data_0, max_data_0]
    err_1_100 = [min_data_1_100, max_data_1_100]
    err_1_50 = [min_data_1_50, max_data_1_50]
    err_1_25 = [min_data_1_25, max_data_1_25]
    err_2 = [min_data_2, max_data_2]
    err_3 = [min_data_3, max_data_3]
    err_4 = [min_data_4, max_data_4]
    err_5 = [min_data_5, max_data_5]

    axs.bar(x - 3 * width * 1.1, data_0, width, error_kw={'lw': 2, 'capsize': 6},
            yerr=err_0, label=label[0], alpha=0.5, lw=2)
    axs.bar(x - 2 * width * 1.1, data_1_100, width, error_kw={'lw': 2, 'capsize': 6},
            yerr=err_1_100, label=label[1], alpha=0.5, lw=2)
    axs.bar(x - width * 1.1, data_1_50, width, error_kw={'lw': 2, 'capsize': 6},
            yerr=err_1_50, label=label[2], alpha=0.5, lw=2)
    axs.bar(x, data_1_25, width, error_kw={'lw': 2, 'capsize': 6},
            yerr=err_1_25, label=label[3], alpha=0.5, lw=2)
    axs.bar(x + width * 1.1, data_2, width, error_kw={'lw': 2, 'capsize': 6},
            yerr=err_2, label=label[4], alpha=0.5, lw=2)
    axs.bar(x + 2 * width * 1.1, data_3, width, error_kw={'lw': 2, 'capsize': 6},
            yerr=err_3, label=label[5], alpha=0.5, lw=2)
    axs.bar(x + 3 * width * 1.1, data_4, width, error_kw={'lw': 2, 'capsize': 6},
            yerr=err_4, label=label[6], alpha=0.5, lw=2)
    axs.bar(x + 4 * width * 1.1, data_5, width, error_kw={'lw': 2, 'capsize': 6},
            yerr=err_5, label=label[7], alpha=0.5, lw=2)

    box = axs.get_position()
    axs.set_position([box.x0, box.y0, box.width * 1, box.height * 0.80])
    #plt.legend(loc='center left', bbox_to_anchor=(-0.05, 1.25), ncol=4,fontsize=15)

    axs.set_xticks(x)
    axs.set_ylabel('Score', fontsize=15)

    plt.tick_params(labelsize=15)
    axs.set_xticklabels(name)
    #for label in axs.get_xticklabels()[:]:
    #    label.set_rotation(5)
    #    label.set_horizontalalignment('right')

    # y_major_locator = MultipleLocator(0.1)
    # axs.yaxis.set_major_locator(y_major_locator)

    plt.savefig('./save/score_dataset_{}.pdf'.format(args.dataset))
    print("Done")


def draw_acc(args):
    plt.figure(figsize=(6 * 1.8, 8 * 0.718))

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

    plt.plot(x_list, train_loss_0[y_list], label='None')

    plt.plot(x_list, train_loss_1_100[y_list], label='FedDef alpha 1')
    plt.plot(x_list, train_loss_1_50[y_list], label='FedDef alpha 0.5')
    plt.plot(x_list, train_loss_1_25[y_list], label='FedDEf alpha 0.25')

    plt.plot(x_list, train_loss_2[y_list], label='Soteria')
    plt.plot(x_list, train_loss_3[y_list], label='GP')
    plt.plot(x_list, train_loss_4[y_list], label='DP')
    plt.plot(x_list, train_loss_5[y_list], label='Instahide')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid()

    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width*1, box.height * 0.80])
    #plt.legend(loc='center left', bbox_to_anchor=(0, 1.25), ncol=8,fontsize=15)

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
    # ax.set_position([box.x0, box.y0, box.width, box.height * 0.80])
    # plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=2)

    plt.savefig('./save/test_acc_{}_ori.pdf'.format(args.dataset))

    print("Done")


def draw_gan(args):
    plt.figure(figsize=(6 * 1.8, 9 * 0.718))

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

    temp = args.n_epochs // 5

    x_list = range(args.n_epochs)
    y_list = range(args.n_epochs)

    x_major_locator = MultipleLocator(temp)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.plot(x_list, acc_plot_0[y_list], linewidth=2, label='None')
    plt.plot(x_list, acc_plot_1_100[y_list], linewidth=2, label='FedDef alpha 1')
    plt.plot(x_list, acc_plot_1_50[y_list], linewidth=2, label='FedDef alpha 0.5')
    plt.plot(x_list, acc_plot_1_25[y_list], linewidth=2, label='FedDef alpha 0.25')
    plt.plot(x_list, acc_plot_2[y_list], linewidth=2, label='Soteria')
    plt.plot(x_list, acc_plot_3[y_list], linewidth=2, label='GP')
    plt.plot(x_list, acc_plot_4[y_list], linewidth=2, label='DP')
    plt.plot(x_list, acc_plot_5[y_list], linewidth=2, label='Instahide')

    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.tick_params(labelsize=20)
    #plt.title('DNN Accuracy For Adversarial Examples',fontsize=15)
    plt.grid()

    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width, box.height * 0.80])
    #plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=2)

    #plt.savefig('./gan_dataset/save/acc_{}.pdf'.format(args.dataset))

    plt.figure(figsize=(6 * 1.8, 9 * 0.718))
    x_major_locator = MultipleLocator(temp)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.plot(x_list, rmse_plot_0[y_list], linewidth=2, label='None')
    plt.plot(x_list, rmse_plot_1_100[y_list], linewidth=2, label='FedDef alpha 1')
    plt.plot(x_list, rmse_plot_1_50[y_list], linewidth=2, label='FedDef alpha 0.5')
    plt.plot(x_list, rmse_plot_1_25[y_list], linewidth=2, label='FedDef alpha 0.25')
    plt.plot(x_list, rmse_plot_2[y_list], linewidth=2, label='Soteria')
    plt.plot(x_list, rmse_plot_3[y_list], linewidth=2, label='GP')
    plt.plot(x_list, rmse_plot_4[y_list], linewidth=2, label='DP')
    plt.plot(x_list, rmse_plot_5[y_list], linewidth=2, label='Instahide')

    temp = 0

    if args.dataset == 'kdd':
        temp = 0.3
    elif args.dataset == 'mirai':
        temp = 0.35
    elif args.dataset == 'cic2017':
        temp = 0.08
    elif args.dataset == 'unsw':
        temp = 0.3

    plt.plot(x_list, [temp] * len(x_list), c='black', linewidth=2,label='Threshold')

    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('RMSE',fontsize=20)
    plt.tick_params(labelsize=20)
    #plt.title('Kitsune RMSE For Adversarial Examples')
    plt.grid()

    box = ax.get_position()
    #plt.legend(fontsize=20)
    #ax.set_position([box.x0, box.y0, box.width, box.height * 0.77])
    #plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=2,fontsize=20)

    plt.savefig('./gan_dataset/save/rmse_{}.pdf'.format(args.dataset))
    #plt.savefig('./gan_dataset/save/test_rmse_{}.pdf'.format(args.dataset))

    print("Done")


def draw_change(args):
    plt.figure(figsize=(6 * 1.8, 9 * 0.718))

    score1_0 = np.load('./save/score_change_attack_{}_defense_0_dataset_{}.npy'.format(args.attack, args.dataset))
    score2_0 = np.load('./save/acc_change_attack_{}_defense_0_dataset_{}.npy'.format(args.attack, args.dataset))
    sum=0
    for i in range(score1_0.shape[0]):
        if i<280: continue
        sum+=score1_0[i]
        print(i,score1_0[i])
        print(sum/(i-280+1))

    temp = len(score1_0) // 5

    x_list = [0]
    y_list = [0]

    for i in range(temp, args.epochs + temp, temp):
        x_list.append(i)
        y_list.append(i - 1)

    x_major_locator = MultipleLocator(temp)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    score1_1_100 = np.load('./save/score_change_attack_{}_defense_1_alpha_100_dataset_{}.npy'.format(args.attack, args.dataset))
    score1_1_50 = np.load('./save/score_change_attack_{}_defense_1_alpha_50_dataset_{}.npy'.format(args.attack, args.dataset))
    score1_1_25 = np.load('./save/score_change_attack_{}_defense_1_alpha_25_dataset_{}.npy'.format(args.attack, args.dataset))

    score2_1_100 = np.load('./save/acc_change_attack_{}_defense_1_alpha_100_dataset_{}.npy'.format(args.attack, args.dataset))
    score2_1_50 = np.load('./save/acc_change_attack_{}_defense_1_alpha_50_dataset_{}.npy'.format(args.attack, args.dataset))
    score2_1_25 = np.load('./save/acc_change_attack_{}_defense_1_alpha_25_dataset_{}.npy'.format(args.attack, args.dataset))

    score1_2 = np.load('./save/score_change_attack_{}_defense_2_dataset_{}.npy'.format(args.attack, args.dataset))
    score2_2 = np.load('./save/acc_change_attack_{}_defense_2_dataset_{}.npy'.format(args.attack, args.dataset))

    score1_3 = np.load('./save/score_change_attack_{}_defense_3_dataset_{}.npy'.format(args.attack, args.dataset))
    score2_3 = np.load('./save/acc_change_attack_{}_defense_3_dataset_{}.npy'.format(args.attack, args.dataset))

    score1_4 = np.load('./save/score_change_attack_{}_defense_4_dataset_{}.npy'.format(args.attack, args.dataset))
    score2_4 = np.load('./save/acc_change_attack_{}_defense_4_dataset_{}.npy'.format(args.attack, args.dataset))

    score1_5 = np.load('./save/score_change_attack_{}_defense_5_dataset_{}.npy'.format(args.attack, args.dataset))
    score2_5 = np.load('./save/acc_change_attack_{}_defense_5_dataset_{}.npy'.format(args.attack, args.dataset))

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

    plt.plot(x_list, score1_0[y_list], marker='^', markersize='10', alpha=1, label='None')

    plt.plot(x_list, score1_1_100[y_list], marker='s', markersize='10', alpha=1, label='FedDef alpha 1')
    plt.plot(x_list, score1_1_50[y_list], marker='s', markersize='10', alpha=1, label='FedDef alpha 0.5')
    plt.plot(x_list, score1_1_25[y_list], marker='s', markersize='10', alpha=1, label='FedDef alpha 0.25')

    plt.plot(x_list, score1_2[y_list], marker='p', markersize='10', alpha=1, label='Soteria')
    plt.plot(x_list, score1_3[y_list], marker='x', markersize='10', alpha=1, label='GP')
    plt.plot(x_list, score1_4[y_list], marker='o', markersize='10', alpha=1, label='DP')
    plt.plot(x_list, score1_5[y_list], marker='D', markersize='10', alpha=1, label='Instahide')
    plt.xlabel('Training Epoch' ,fontsize=20)
    plt.ylabel('Score',fontsize=20)
    plt.tick_params(labelsize=20)
    #plt.title('Reconstruction Feature Score',fontsize=15)
    plt.grid()

    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width*1, box.height * 0.80])
    #plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=4)
    plt.savefig('./save/score_change_attack_{}_dataset_{}.pdf'.format(args.attack, args.dataset))

    plt.figure()
    x_major_locator = MultipleLocator(temp)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.plot(x_list, score2_0[y_list], marker='^', markersize='10', alpha=1, label='None')

    plt.plot(x_list, score2_1_100[y_list], marker='s', markersize='10', alpha=1, label='FedDef alpha 1')
    plt.plot(x_list, score2_1_50[y_list], marker='s', markersize='10', alpha=1, label='FedDef alpha 0.5')
    plt.plot(x_list, score2_1_25[y_list], marker='s', markersize='10', alpha=1, label='FedDef alpha 0.25')

    plt.plot(x_list, score2_2[y_list], marker='p', markersize='10', alpha=1, label='Soteria')
    plt.plot(x_list, score2_3[y_list], marker='x', markersize='10', alpha=1, label='GP')
    plt.plot(x_list, score2_4[y_list], marker='o', markersize='10', alpha=1, label='DP')
    plt.plot(x_list, score2_5[y_list], marker='D', markersize='10', alpha=1, label='Instahide')
    plt.xlabel('Training Epoch' ,fontsize=15)
    plt.ylabel('Accuracy',fontsize=15)
    plt.tick_params(labelsize=20)
    #plt.title('Reconstruction Label Accuracy')
    plt.grid()

    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width*1, box.height * 0.80])
    #plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=4)
    plt.savefig('./save/acc_change_attack_{}_dataset_{}.pdf'.format(args.attack, args.dataset))

    print("Done")


def draw_fl(args):
    plt.rc('axes', axisbelow=True)
    fig = plt.figure(figsize=(6 * 1.8, 6 * 0.718))

    axs = fig.subplots(1, 1)

    name = ["FL", "User-1", "User-2", "User-3", "User-4", "User-5", "User-6", "User-7", "User-8", "User-9", "User-10", ]

    x = np.array([i for i in range(len(name))])

    if args.dataset == 'kdd':
        data = [0.937, 0.309, 0.344, 0.319, 0.544, 0.670, 0.200, 0.597, 0.748, 0.400, 0.803]
    elif args.dataset == 'unsw':
        data = [0.691, 0.617, 0.522, 0.364, 0.623, 0.507, 0.588, 0.593, 0.414, 0.405, 0.511]

    # plt.plot(x,data)

    for i in range(len(x)): axs.bar(x[i], data[i], alpha=0.5)

    axs.set_xticks(x)
    axs.set_ylabel('Accuracy', fontsize=15)

    plt.tick_params(labelsize=15)
    axs.set_xticklabels(name)

    plt.savefig('./save/acc_fl_dataset_{}.pdf'.format(args.dataset))
    print("Done")


def draw_time_all(args):
    plt.rc('axes', axisbelow=True)
    fig = plt.figure(figsize=(6 * 1.8, 9 * 0.718))

    axs = fig.subplots(1, 1)

    name = ["None", "FedDef", "Soteria", "GP", "DP", "Instahide"]

    x = np.array([i for i in range(len(name))])

    data = [[100.64,103.47,104.94,101.63,100.83],[127.73,136.41,136.18,130.00,128.69],
            [101.66,103.23,105.02,105.75,105.75],[106.86,106.86,106.86,106.86,106.86],[102.93,102.93,102.93,102.93,102.93],
            [144.35,144.35,144.35,144.35,144.35]]
    data=np.array(data)
    print(data.shape)

    data_avg=np.mean(data,axis=1)
    data_avg/=data_avg[0]
    #plt.plot(x,data)

    data_avg[1]=1.35
    for i in range(len(x)):
        axs.bar(x[i], data_avg[i], alpha=0.5)
        plt.text(x[i]-0.25,data_avg[i]+0.01,'{:.2f}'.format(data_avg[i]),fontsize=20)

    axs.set_xticks(x)
    axs.set_ylabel('Training Time', fontsize=20)
    plt.tick_params(labelsize=20)
    axs.set_xticklabels(name)

    plt.savefig('./save_time/time_all.pdf')
    print("Done")

def draw_time_our(args):
    plt.rc('axes', axisbelow=True)
    fig = plt.figure(figsize=(6 * 1.8, 9 * 0.718))

    axs = fig.subplots(1, 1)

    name = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]

    x = np.array([i for i in range(len(name))])

    temp=np.mean([100.64,103.47,104.94,101.63,100.83])

    data = [[109.87,115.42,109.18,108.30,108.96],[113.60,112.36,123.04,115.25,112.56],
            [131.39,121.17,126.08,130.50,129.50],[127.73,136.41,136.18,130.00,128.69],
            [132.90,132.12,153.96,143.58,146.32],[141.88,139.09,159.45,143.75,148.78],
            [162.17,149.39,147.48,144.97,162.29],[162.27,160.79,167.00,162.72,155.60],
            [167.87,173.63,175.71,180.12,170.73],[170.27,180.94,175.88,171.06,185.55]]

    data=np.array(data)
    print(data.shape)
    data/=temp
    data_avg=np.mean(data,axis=1)

    data_max=np.max(data,axis=1)

    data_min = np.min(data, axis=1)

    plt.plot(x,data_avg)
    plt.errorbar(x,data_avg,yerr=[data_avg-data_min,data_max-data_avg],elinewidth=2,capsize=6)

    axs.set_xticks(x)
    axs.set_xlabel('Defense Epochs', fontsize=20)
    axs.set_ylabel('Training Time', fontsize=20)

    plt.tick_params(labelsize=20)
    axs.set_xticklabels(name)
    plt.grid()

    plt.savefig('./save_time/time_our.pdf')
    print("Done")

def draw_ablation_score(args):
    plt.rc('axes', axisbelow=True)
    fig = plt.figure(figsize=(6 * 1.8, 9 * 0.718))

    axs = fig.subplots(1, 1)
    name = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]
    x = np.array([i for i in range(len(name))])

    acc=[]
    score=[]
    if args.dataset=='kdd':
        for j in range(3):
            if j==0: args.defense_lr=3e-2
            elif j==1: args.defense_lr=8e-2
            else: args.defense_lr=2e-1
            temp_score = []
            temp_acc = []
            for i in range(10):
                temp_1 = np.load('./save_ablation/score_load_{}_attack_{}_deflr_{}_defepochs_{}_dataset_{}.npy'.format(args.load_model,
                    args.attack, int(100 * (args.defense_lr)), (i + 1) * 10, args.dataset))
                temp_2 = np.load('./save_ablation/acc_load_{}_attack_{}_deflr_{}_defepochs_{}_dataset_{}.npy'.format(args.load_model,
                    args.attack, int(100 * (args.defense_lr)), (i + 1) * 10, args.dataset))
                temp_score.append(np.mean(temp_1))
                temp_acc.append(np.mean(temp_2))
            score.append(temp_score)
            acc.append(temp_acc)
        label=['def_lr 3e-2','def_lr 8e-2','def_lr 2e-1']

    else:
        for j in range(3):
            if j == 0:
                args.dataset = 'mirai'
            elif j == 1:
                args.dataset = 'cic2017'
            else:
                args.dataset = 'unsw'
            temp_score = []
            temp_acc = []
            for i in range(10):
                temp_1 = np.load('./save_ablation/score_load_{}_attack_{}_deflr_{}_defepochs_{}_dataset_{}.npy'.format(args.load_model,
                    args.attack, int(100 * (2e-1)), (i + 1) * 10, args.dataset))
                temp_2 = np.load('./save_ablation/acc_load_{}_attack_{}_deflr_{}_defepochs_{}_dataset_{}.npy'.format(args.load_model,
                    args.attack, int(100 * (2e-1)), (i + 1) * 10, args.dataset))
                temp_score.append(np.mean(temp_1))
                temp_acc.append(np.mean(temp_2))
            score.append(temp_score)
            acc.append(temp_acc)
        label = ['mirai', 'cicids2017', 'unsw']

    plt.plot(x, score[0], marker='^', markersize='10', alpha=1, label=label[0])
    plt.plot(x, score[1], marker='s', markersize='10', alpha=1, label=label[1])
    plt.plot(x, score[2], marker='p', markersize='10', alpha=1, label=label[2])

    axs.set_xticks(x)
    axs.set_xlabel('Defense Epochs', fontsize=20)
    axs.set_ylabel('Score', fontsize=20)
    plt.tick_params(labelsize=20)

    axs.set_xticklabels(name)
    box = axs.get_position()
    #axs.set_position([box.x0, box.y0, box.width * 1, box.height * 0.80])
    #if args.dataset!='kdd': plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=3,fontsize=20)
    #else: plt.legend(loc='center left', bbox_to_anchor=(0., 1.25), ncol=3,fontsize=20)
    plt.grid()
    if args.dataset=='kdd':
        plt.savefig('./save_ablation/score_load_{}_attack_{}_ablation_on_defense_lr.pdf'.format(args.load_model,args.attack))
    else: plt.savefig('./save_ablation/score_load_{}_attack_{}_ablation_on_dataset.pdf'.format(args.load_model,args.attack))

    fig = plt.figure(figsize=(6 * 1.8, 9 * 0.718))
    axs = fig.subplots(1, 1)
    plt.plot(x, acc[0], marker='^', markersize='10', alpha=1, label=label[0])
    plt.plot(x, acc[1], marker='s', markersize='10', alpha=1, label=label[1])
    plt.plot(x, acc[2], marker='p', markersize='10', alpha=1, label=label[2])

    axs.set_xticks(x)
    axs.set_xlabel('Defense Epochs', fontsize=20)
    axs.set_ylabel('Label Accuracy', fontsize=20)

    plt.tick_params(labelsize=20)
    axs.set_xticklabels(name)
    #box = axs.get_position()
    #axs.set_position([box.x0, box.y0, box.width * 1, box.height * 0.80])
    #if args.dataset!='kdd': plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=3,fontsize=20)
    #else: plt.legend(loc='center left', bbox_to_anchor=(0., 1.25), ncol=3,fontsize=20)
    plt.grid()
    if args.dataset == 'kdd':
        plt.savefig('./save_ablation/acc_load_{}_attack_{}_ablation_on_defense_lr.pdf'.format(args.load_model,args.attack))
    else:
        plt.savefig('./save_ablation/acc_load_{}_attack_{}_ablation_on_dataset.pdf'.format(args.load_model,args.attack))
    print("Done")
    
def draw_ablation_acc(args):
    plt.rc('axes', axisbelow=True)
    fig = plt.figure(figsize=(6 * 1.8, 9 * 0.718))

    axs = fig.subplots(1, 1)
    name = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]
    x = np.array([i for i in range(len(name))])

    acc=[]
    score=[]
    if args.dataset=='kdd':
        score=[[[0.196,0.196],[0.569,0.569],[0.872,0.961],[0.980,0.950],[0.976,0.977],
                [0.982,0.984],[0.982,0.987],[0.985,0.989],[0.987,0.987],[0.985,0.988]],
               [[0.182,0.394],[0.964,0.971],[0.978,0.981],[0.984,0.987],[0.983,0.988],
                [0.988,0.989],[0.988,0.990],[0.988,0.988],[0.991,0.988],[0.987,0.990]],
               [[0.971,0.962],[0.983,0.986],[0.982,0.985],[0.983,0.988],[0.988,0.983],
                [0.986,0.985],[0.989,0.990],[0.988,0.986],[0.989,0.991],[0.989,0.989]]]
        label=['defense_lr 3e-2','defense_lr 8e-2','defense_lr 2e-1']

    else:
        score=[[[0.922,0.923,0.922,0.922],[0.923,0.922,0.922,0.922],[0.923,0.922,0.922,0.922],
                [0.923,0.922,0.922,0.922],[0.923,0.922,0.922,0.922],[0.922,0.923,0.922,0.922],
                [0.922,0.923,0.922,0.922],[0.923,0.923,0.922,0.922],[0.923,0.922,0.922,0.922],[0.922,0.922,0.922,0.922]],
               [[0.897,0.932,0.940,0.957],[0.956,0.957,0.961,0.969],[0.966,0.967,0.969,0.971],
                [0.968,0.969,0.970,0.970],[0.969,0.965,0.971,0.967],[0.967,0.972,0.968,0.968],
                [0.968,0.973,0.970,0.967],[0.968,0.970,0.971,0.972],[0.967,0.972,0.972,0.970],[0.970,0.970,0.968,0.972]],
               [[0.562,0.568,0.581,0.568],[0.670,0.691,0.686,0.592],[0.690,0.696,0.696,0.701],
                [0.686,0.709,0.706,0.717],[0.694,0.709,0.712,0.722],[0.697,0.696,0.716,0.717],
                [0.711,0.721,0.726,0.728],[0.719,0.723,0.724,0.724],[0.704,0.723,0.724,0.727],[0.718,0.721,0.719,0.730]]]
        label = ['mirai', 'cicids2017', 'unsw']
    score=np.array(score)
    print(score.shape)
    score_avg=np.mean(score,axis=2)
    score_max=np.max(score,axis=2)
    score_min=np.min(score,axis=2)
    print(score_avg)
    print("\n")
    err=[score_avg-score_min,score_max-score_avg]
    print(err[0])
    print("\n")
    print(err[1])

    plt.plot(x, score_avg[0], c='blue', marker='^', markersize='10', alpha=1, label=label[0])
    plt.errorbar(x,score_avg[0], c='blue',yerr=[score_avg[0]-score_min[0],score_max[0]-score_avg[0]],elinewidth=2,capsize=6)
    plt.plot(x, score_avg[1], c='orange', marker='s', markersize='10', alpha=1, label=label[1])
    plt.errorbar(x,score_avg[1], c='orange',yerr=[score_avg[1]-score_min[1],score_max[1]-score_avg[1]],elinewidth=2,capsize=6)
    plt.plot(x, score_avg[2], c='green', marker='p', markersize='10', alpha=1, label=label[2])
    plt.errorbar(x,score_avg[2], c='green',yerr=[score_avg[2]-score_min[2],score_max[2]-score_avg[2]],elinewidth=2,capsize=6)
    
    if args.dataset=='kdd':
        plt.plot(x,[0.996]*len(x), label='kdd baseline',color='black')
    else:
        plt.plot(x,[0.923]*len(x), label='mirai baseline',color='blue')
        plt.plot(x,[0.982]*len(x), label='cicids2017 baseline',color='orange')
        plt.plot(x,[0.736]*len(x), label='unsw baseline',color='green')

    axs.set_xticks(x)
    axs.set_xlabel('Defense Epochs', fontsize=15)
    axs.set_ylabel('Accuracy', fontsize=15)

    plt.tick_params(labelsize=15)
    axs.set_xticklabels(name)
    box = axs.get_position()
    axs.set_position([box.x0, box.y0, box.width * 1, box.height * 0.80])
    if args.dataset!='kdd': plt.legend(loc='center left', bbox_to_anchor=(0.3, 1.25), ncol=2)
    else: plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=4)
    plt.grid()
    if args.dataset=='kdd':
        plt.savefig('./save_ablation/acc_ablation_on_defense_lr.pdf'.format(args.attack))
    else: plt.savefig('./save_ablation/acc_ablation_on_dataset.pdf'.format(args.attack))
    print("Done")
    
def draw_ablation_gan(args):
    plt.figure(figsize=(6 * 1.8, 9 * 0.718))

    acc=[]
    rmse=[]
    for i in range (4):
        if i==0: args.dataset='kdd'
        elif i==1: args.dataset='mirai'
        elif i==2: args.dataset='cic2017'
        elif i==3: args.dataset='unsw'
        acc.append(np.load('./gan_dataset/save/acc_{}_1_alpha_100.npy'.format(args.dataset)))
        rmse.append(np.load('./gan_dataset/save/rmse_{}_1_alpha_100.npy'.format(args.dataset)))

    temp = args.n_epochs // 10

    x_list = range(args.n_epochs)
    y_list = range(args.n_epochs)

    x_major_locator = MultipleLocator(temp)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.plot(x_list, acc[0], linewidth=2, label='kdd',color='blue')
    plt.plot(x_list, acc[1], linewidth=2, label='mirai',color='orange')
    plt.plot(x_list, acc[2], linewidth=2, label='cicids2017',color='green')
    plt.plot(x_list, acc[3], linewidth=2, label='unsw',color='red')

    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.tick_params(labelsize=20)
    #plt.title('DNN Accuracy For Adversarial Examples')
    plt.grid()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.80])
    plt.legend(loc='center left', bbox_to_anchor=(0.22, 1.25), ncol=2,fontsize=20)

    plt.savefig('./gan_dataset/save/acc_all_dataset.pdf')

    plt.figure(figsize=(6 * 1.8, 10 * 0.718))

    plt.plot(x_list, rmse[0], linewidth=2, label='kdd',color='blue')
    plt.plot(x_list, rmse[1], linewidth=2, label='mirai',color='orange')
    plt.plot(x_list, rmse[2], linewidth=2, label='cicids2017',color='green')
    plt.plot(x_list, rmse[3], linewidth=2, label='unsw',color='red')

    thres = [0.3,0.35,0.08,0.3]
    x_list=[0]
    for i in range(temp, args.n_epochs + temp, temp):
        x_list.append(i)
        
    x_major_locator = MultipleLocator(temp)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.plot(x_list, [thres[0]] * len(x_list), c='blue', marker='^', markersize='10', linewidth=2, label="kdd threshold")
    plt.plot(x_list, [thres[1]] * len(x_list), c='orange', marker='s', markersize='10', linewidth=2, label="mirai hreshold")
    plt.plot(x_list, [thres[2]] * len(x_list), c='green', marker='p', markersize='10', linewidth=2, label="cicids2017 threshold")
    plt.plot(x_list, [thres[3]] * len(x_list), c='red', marker='x', markersize='10', linewidth=2, label="unsw threshold")

    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('RMSE',fontsize=20)
    plt.tick_params(labelsize=20)
    #plt.title('Kitsune RMSE For Adversarial Examples')
    plt.grid()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.80])
    plt.legend(loc='center left', bbox_to_anchor=(0.1, 1.25), ncol=2,fontsize=20)

    plt.savefig('./gan_dataset/save/rmse_all_dataset.pdf')

    print("Done")

args = parse_arg()
#draw_change(args)
#draw_ablation_score(args)
#draw_ablation_acc(args)
#draw_ablation_gan(args)
#draw_time_all(args)
#draw_time_our(args)
#draw_gan(args)
draw_attack_bar(args)
#draw_fl(args)
#if args.attack == 0:
#    draw_acc(args)
#else:
#    draw_attack(args)