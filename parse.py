import argparse

def parse_arg():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('-epochs', type=int, default=300, help="rounds of training")
    
    parser.add_argument('-num_users', type=int, default=1, help="number of users: K")
    
    parser.add_argument('-local_ep', type=int, default=1, help="the number of local epochs: E")
    
    parser.add_argument('-local_bs', type=int, default=1, help="local batch size: B")
    
    parser.add_argument('-lr', type=float, default=3e-3, help="learning rate")
    
    parser.add_argument('-eval', type=int, default=1, help="whether to eval")
    
    parser.add_argument('-iid', type=int, default=1, help="use iid or non-iid")
    
    parser.add_argument('-save_change', type=int, default=0, help="save reconstruction performance change plot")
    
    parser.add_argument('-save_ablation', type=int, default=0, help="save abalation results")

    # model arguments
    parser.add_argument('-model', type=str, default='dnn', help='model name')
    
    parser.add_argument('-device', type=str, default='cuda:0', help='cuda device')
    
    parser.add_argument('-decay', type=float, default=0.9, help='decay')
    
    parser.add_argument('-decay_epochs', type=int, default=20, help="decay epochs")
    
    parser.add_argument('-test_num', type=int, default=-1, help='test cases')
    
    # other arguments
    parser.add_argument('-dataset', type=str, default='kdd', help="name of datasetnum")
    
    parser.add_argument('-recon_epochs', type=int, default=300, help="reconstruct epoch per batch")
    
    parser.add_argument('-attack', type=int, default=0, help="deploy attack")
    
    parser.add_argument('-defense', type=int, default=0, help="deploy defense")
    
    parser.add_argument('-alpha', type=float, default=1, help="our defense parameter for distance between gradients")
    
    parser.add_argument('-defense_epochs', type=int, default=40, help="defense epoch")
    
    parser.add_argument('-defense_lr', type=float, default=2e-1, help="defense learning rate")
    
    parser.add_argument('-g_value', type=float, default=1e-15, help="gradient constraint")
    
    parser.add_argument('-max_dis', type=float, default=1, help="max dis")
    
    parser.add_argument('-load_model', type=int, default=0, help="load trained model")
    
    parser.add_argument('-save_model', type=int, default=0, help="save trained model")
    
    parser.add_argument('-save_plot', type=int, default=0, help="save plot")
    
    parser.add_argument('-save_recon', type=int, default=0, help="save recon x and y")
    
    parser.add_argument("-klam",default=4, type=int, help="How many images to mix with")
    
    parser.add_argument('-upper', default=0.65, type=float, help='the upper bound of any coefficient')
    
    # GAN argumentss
    parser.add_argument("-n_epochs", type=int, default=300, help="number of epochs of training")
    parser.add_argument("-gan_lr", type=float, default=1e-4, help="gan lr")
    parser.add_argument("-b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("-b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("-clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("-n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
    parser.add_argument("-pretrain", type=int, default=0, help="pretrain discriminator")
    parser.add_argument("-use_ori", type=int, default=0, help="use original datset instead of GAN dataset")
    
    
    args = parser.parse_args()
    
    return args