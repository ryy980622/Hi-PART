import argparse
import random
import time
import torch
import numpy as np
from network import GNet
from trainer import Trainer
from utils.data_loader import FileLoader
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import sys
sys.path.append("..")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=3, help='seed')
    parser.add_argument('-data', default='MUTAG', help='data folder name')
    parser.add_argument('-fold', type=int, default=2, help='fold (1..10)')
    parser.add_argument('-num_epochs', type=int, default=300, help='epochs')
    parser.add_argument('-batch', type=int, default=16, help='batch size')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-l_num', type=int, default=3, help='layer num')
    parser.add_argument('-h_dim', type=int, default=128, help='hidden dim')
    parser.add_argument('-deg_as_tag', type=int, default=1, help='1 or degree')
    parser.add_argument('-l_dim', type=int, default=128, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0.3, help='drop net')
    parser.add_argument('-drop_c', type=float, default=0.3, help='drop output')
    parser.add_argument('-act_n', type=str, default='LeakyReLU', help='network act')
    parser.add_argument('-act_c', type=str, default='ELU', help='output act')
    parser.add_argument('-ks', nargs='+', type=float, default=[0.7, 0.8, 0.9])
    parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    args, _ = parser.parse_known_args()
    return args

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def app_run(args, G_data, fold_idx):
    G_data.use_fold_data(fold_idx)
    net = GNet(G_data.feat_dim, G_data.num_class, args)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))

    net.to(device)
    trainer = Trainer(args, net, G_data)
    max_acc, loss_train, train_accs = trainer.train()
    return max_acc, loss_train, train_accs

def plot(loss, dataset):
    x = range(0, len(loss))
    x = np.array(x)
    loss = np.array(loss)
    x_smooth = np.linspace(x.min(), x.max(), 300)  # list没有min()功能调用
    y_smooth = make_interp_spline(x, loss)(x_smooth)
    plt.plot(x_smooth, y_smooth, '-')
    plt_title = dataset
    plt.title(plt_title)
    plt.xlabel('Epoch')
    plt.ylabel('LOSS')
    # plt.savefig(file_name)
    np.savetxt('loss/'+dataset+'.txt', loss)
    plt.show()
def main():
    args = get_args()
    print(args)
    set_random(args.seed)
    start = time.time()
    G_data = FileLoader(args).load_data()
    print('load data using ------>', time.time()-start)
    if args.fold == 0:
        sum_acc = 0
        for fold_idx in range(10):
            print('start training ------> fold', fold_idx+1)
            acc, loss_train, train_accs = app_run(args, G_data, fold_idx)
            sum_acc += acc
        print('average accuracy:', sum_acc/10)
    else:
        print('start training ------> fold', args.fold)
        acc, loss_train, train_accs = app_run(args, G_data, args.fold-1)

if __name__ == "__main__":
    main()
