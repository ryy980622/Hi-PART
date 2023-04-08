import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ops import GCN,  Initializer, norm_g,GraphAttentionLayer,SGC
#from utils.GAT_Layer import  GraphAttentionLayer
from utils.GIN import GIN

class GNet(nn.Module):
    def __init__(self, in_dim, n_classes, args):
        super(GNet, self).__init__()
        self.n_act = getattr(nn, args.act_n)() #elu激活函数
        self.c_act = getattr(nn, args.act_c)() #elu激活函数

        self.ori_gcn1 = GCN(in_dim, args.l_dim, self.n_act, args.drop_n)
        # self.ori_gcn1 = GCN(in_dim,n_classes, self.n_act, args.drop_n)
        # self.ori_gcn2 = GCN(args.l_dim, args.l_dim, self.n_act, args.drop_n)
        # self.ori_gcn3 = GCN(args.l_dim, args.l_dim, self.n_act, args.drop_n)
        # self.ori_sgc = SGC(in_dim, args.l_dim)
        # self.sgc = SGC(args.l_dim, 3*args.l_dim)

        # self.gat1= GraphAttentionLayer(args.l_dim,args.l_dim,dropout=args.drop_n,alpha=0.2)
        # self.gat2 = GraphAttentionLayer(args.l_dim, 2*args.l_dim, dropout=args.drop_n, alpha=0.2)
        # self.gat3 = GraphAttentionLayer(2*args.l_dim, 3 * args.l_dim, dropout=args.drop_n, alpha=0.2)

        # self.ori_gin1=GIN(in_dim, 2*args.l_dim, args.l_dim, args.drop_n)
        # self.ori_gin2 = GIN(args.l_dim, 2*args.l_dim, args.l_dim, args.drop_n)
        # self.ori_gin3 = GIN(args.l_dim, 2*args.l_dim, args.l_dim, args.drop_n)
        # self.gin1=GIN(args.l_dim, 2*args.l_dim, 2*args.l_dim, args.drop_n)
        # self.gin2 = GIN(2*args.l_dim, 3 * args.l_dim,3 * args.l_dim, args.drop_n)
        # self.gin3 = GIN(3 * args.l_dim, 4 * args.l_dim, 4 * args.l_dim, args.drop_n)
        # self.gin4 = GIN(4 * args.l_dim, 4 * args.l_dim, 4 * args.l_dim, args.drop_n)

        # self.gcn1 = GCN(in_dim, 2 * args.l_dim, self.n_act, args.drop_n)
        self.gcn1 = GCN(args.l_dim, args.l_dim, self.n_act, args.drop_n)
        self.gcn2 = GCN(args.l_dim, 2 * args.l_dim, self.n_act, args.drop_n)
        self.gcn3 = GCN(2 * args.l_dim, 2 * args.l_dim, self.n_act, args.drop_n)
        self.gcn4 = GCN(3 * args.l_dim, 3 * args.l_dim, self.n_act, args.drop_n)

        self.out_l_1 = nn.Linear(2 * args.l_dim, 3 * args.l_dim)  # readout后的全连接层
        self.out_l_2 = nn.Linear(3 * args.l_dim, args.l_dim)  # readout后的全连接层
        self.out_l_3 = nn.Linear(args.l_dim, n_classes)  # 预测的全连接层



        self.out_drop = nn.Dropout(p=args.drop_c)
        self.out = nn.Linear(3* args.l_dim, n_classes)


        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Initializer.weights_init(self)

    def forward(self, gs, hs, labels,ori_gs,ori_hs):
        hs = self.embed(gs, hs,ori_gs,ori_hs)
        logits=self.classify(hs)
        #logits = self.classify(hs)  #logits:8*2
        #print(logits.shape,labels.shape)
        return self.metric(logits, labels)

    def embed(self, gs, hs, ori_gs, ori_hs):
        o_hs = []
        for g, h,ori_g,ori_h in zip(gs, hs,ori_gs, ori_hs):
            h = self.embed_one(g, h, ori_g, ori_h)   #g,h是每张图的邻接矩阵和node embedding
            #h = self.embed_one_readout(g, h, ori_g, ori_h)
            #h = self.embed_one_decouple(g, h, ori_g, ori_h)
            o_hs.append(h) #h是每张图的embedding
        hs = torch.stack(o_hs, 0)   #8*576
        return hs

    def embed_one(self, g, h, ori_g,ori_h):
        #g[0]=torch.ones(1,g.shape[0])
        g = norm_g(g)  #对g做度的归一化
        ori_g = norm_g(ori_g)
        ori_h = self.ori_gcn1(ori_g, ori_h)
        #ori_h = self.ori_gcn2(ori_g, ori_h)
        #ori_h = self.ori_gcn3(ori_g, ori_h)
        #ori_h = self.ori_gcn2(ori_g, ori_h)
        #ori_h = self.ori_gcn3(ori_g, ori_h)
        h=torch.zeros(g.shape[0], ori_h.shape[1]).to(self.device)
        h[1:ori_h.shape[0]+1, ]=ori_h
        h = self.gcn1(g, h)  #通过gcn将node embedding的维度:84->48
        h = self.gcn2(g, h)
        h = self.gcn3(g, h)

        # h = self.gcn4(g, h)
        # h = self.gcn5(g, h)
        # print("h:",h.shape)
        return h[0]
    def embed_one_GIN(self, g, h, ori_g,ori_h):
        g = norm_g(g)  # 对g做度的归一化
        ori_g = norm_g(ori_g)
        ori_h = self.ori_gin1(ori_g, ori_h)
        ori_h = self.ori_gin2(ori_g, ori_h)
        ori_h = self.ori_gin3(ori_g, ori_h)

        #ori_h = self.ori_gcn2(ori_g, ori_h)
        # ori_h = self.ori_gcn3(ori_g, ori_h)
        return self.readout(ori_h)
    def embed_one_decouple(self, g, h, ori_g, ori_h):
        # g[0]=torch.ones(1,g.shape[0])
        g = norm_g(g)  # 对g做度的归一化
        ori_g = norm_g(ori_g)
        g=torch.matmul(torch.matmul(g,g),g)
        ori_g = torch.matmul(ori_g,ori_g)
        ori_h = self.ori_sgc(torch.matmul(ori_g,ori_h))
        #ori_h = self.ori_gcn1(ori_g, ori_h)
        # ori_h = self.ori_gcn2(ori_g, ori_h)
        # ori_h = self.ori_gcn3(ori_g, ori_h)
        h = torch.zeros(g.shape[0], ori_h.shape[1])
        h[1:ori_h.shape[0] + 1, :] = ori_h
        h = self.sgc(torch.matmul(g,h))
        #h = self.gcn1(g, h)  # 通过gcn将node embedding的维度:84->48
        #h = self.gcn2(g, h)
        #h = self.gcn3(g, h)

        # h = self.gcn4(g, h)
        # h = self.gcn5(g, h)
        # print("h:",h.shape)
        return h[0]
    def embed_one_readout(self, g, h, ori_g,ori_h):
        g = norm_g(g)  # 对g做度的归一化
        ori_g = norm_g(ori_g)
        ori_h = self.ori_gcn1(ori_g, ori_h)
        #ori_h = self.ori_gcn2(ori_g, ori_h)
        # ori_h = self.ori_gcn3(ori_g, ori_h)
        return self.readout(ori_h)
    def readout(self, hs):
        h_max = torch.max(hs,0)[0] #48维每一维都取最大值
        h_sum = torch.sum(hs, 0) #48维每一维都取和
        h_mean = torch.mean(hs, 0) #48维每一维都取平均值
        h = torch.cat([h_max, h_sum, h_mean])
        #h= torch.cat([h_max, h_max, h_max])
        #return h #h:576=48*4*3
        return h_max

    def classify(self, h):  #得到每个图的embedding之后的全连接层分类器

        #h = self.out_drop(h)
        #h=self.out(h)
        #h = self.c_act(h)
        #return F.log_softmax(h, dim=1)
        h = self.out_l_1(h)
        h = self.c_act(h)
        h = self.out_drop(h)
        h = self.out_l_2(h)
        h = self.c_act(h)
        h = self.out_drop(h)
        h = self.out_l_3(h)
        return F.log_softmax(h, dim=1)




    def metric(self, logits, labels):
        loss = F.nll_loss(logits, labels)
        _, preds = torch.max(logits, 1)
        acc = torch.mean((preds == labels).float())
        return loss, acc
