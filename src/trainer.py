import torch
from tqdm import tqdm
import torch.optim as optim
from utils.dataset import GraphData
import numpy as np
import random
import time
from torch.optim import lr_scheduler

class Trainer:
    def __init__(self, args, net, G_data):
        self.args = args
        self.net = net
        #self.net = net.cuda()
        self.feat_dim = G_data.feat_dim
        self.fold_idx = G_data.fold_idx
        self.init(args, G_data.train_gs, G_data.test_gs)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def init(self, args, train_gs, test_gs):
        print('#train: %d, #test: %d' % (len(train_gs), len(test_gs)))
        #random.shuffle(train_gs)
        #val_gs = train_gs[:len(train_gs)//10]
        #train_gs = train_gs[len(train_gs)//10:]
        train_data = GraphData(train_gs, self.feat_dim)
        #val_data = GraphData(val_gs, self.feat_dim)
        test_data = GraphData(test_gs, self.feat_dim)
        self.train_d = train_data.loader(self.args.batch, True)
        self.test_d = test_data.loader(self.args.batch, False)
        #self.val_d = val_data.loader(self.args.batch, False)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.args.lr, amsgrad=True,
            weight_decay=0.0008)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=1)

    def to_cuda(self, gs):
        if torch.cuda.is_available():
            if type(gs) == list:
                return [g.cuda() for g in gs]
            return gs.cuda()
        return gs

    def run_epoch(self, epoch, data, model, optimizer, scheduler):
        losses, accs, n_samples = [], [], 0
        model=model.to(self.device)
        start = time.clock()
        for batch in tqdm(data, desc=str(epoch), unit='b'):
            cur_len, gs, hs, ys, ori_gs, ori_hs = batch


            gs, hs, ys, ori_gs , ori_hs= map(self.to_cuda, [gs, hs, ys, ori_gs, ori_hs])  #gs是邻接矩阵的列表，hs是每个图点特征向量的列表，ys是每个图的标签
            loss, acc = model(gs, hs, ys, ori_gs, ori_hs)
            losses.append(loss*cur_len)
            accs.append(acc*cur_len)
            n_samples += cur_len
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if scheduler is not None:
            scheduler.step()
            lr = scheduler.get_lr()
            # print('learning rate:', lr)
        end = time.clock()
        time_epoch = (end - start)*500
        # print('Running time: %s Seconds' % time_epoch)
        avg_loss, avg_acc = sum(losses) / n_samples, sum(accs) / n_samples
        return avg_loss.item(), avg_acc.item()

    def train(self):
        max_acc = 0.0
        max_acc_val=0.0
        val_test=0.0
        train_str = 'Train epoch %d: loss %.5f acc %.5f'
        test_str = 'Test epoch %d: loss %.5f acc %.5f max %.5f'
        val_str='Val epoch %d: loss %.5f acc %.5f max_test %.5f'
        line_str = '%d:\t%.5f\n'
        loss_train=[]
        train_accs = []
        for e_id in range(self.args.num_epochs):
            self.net.train()
            loss, acc = self.run_epoch(
                e_id, self.train_d, self.net, self.optimizer, self.scheduler)

            loss_train.append(loss)

            with torch.no_grad():
                self.net.eval()
                loss, acc = self.run_epoch(e_id, self.test_d, self.net, None, None)
                loss_t, acc_train = self.run_epoch(e_id, self.train_d, self.net, None,None)
                # print(train_str % (e_id, loss_t, acc_train))
                train_accs.append(acc_train)
                #loss_val, acc_val = self.run_epoch(e_id, self.val_d, self.net, None)
            max_acc = max(max_acc, acc)
            '''
            if acc_val>max_acc_val:
                max_acc_val=acc_val
                val_test=acc
            elif  acc_val==max_acc_val:
                val_test = max(acc, val_test)
            print(val_str % (e_id, loss_val, acc_val, val_test))
            '''
            print(test_str % (e_id, loss, acc, max_acc))
            #print("train_acc:", acc_train)

        with open(self.args.acc_file, 'a+') as f:
            f.write(line_str % (self.fold_idx, max_acc))

        return max_acc, loss_train, train_accs