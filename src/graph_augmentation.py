import math
import numpy as np
from queue import Queue, PriorityQueue
import time


import networkx as nx
import pymysql


def read_file(edges, degree, g_dict, connected_fields):
    w = []
    edge = {}
    visit = {}
    cnt = 1
    sum = 0
    n = 0
    m = 0
    for item in edges:
        a = item[0]
        b = item[1]
        if a not in connected_fields or b not in connected_fields:
            continue
        m += 1
        g_dict[(a, b)] = 1
        g_dict[(b, a)] = 1
        sum += 1
        if a not in degree.keys():
            degree[a] = 1
            edge[a] = [b]
        else:
            degree[a] += 1
            edge[a].append(b)
        if b not in degree.keys():
            degree[b] = 1
            edge[b] = [a]
        else:
            degree[b] += 1
            edge[b].append(a)
    n = len(degree)
    return n, m, degree, g_dict, sum, edge

def dfs(pos,id2tag,id2size,id2child):
    if len(id2child[pos])==1:
        return id2tag[pos],1,id2tag,id2size  #返回叶子结点标签和大小
    size=0
    tag=0
    tag2num={}
    mx=0
    for child in id2child[pos]:
        child_tag,child_size,id2tag,id2size=dfs(child,id2tag,id2size,id2child)
        if child_tag not in tag2num.keys():
            tag2num[child_tag]=1
        else:
            tag2num[child_tag]+=1
        if tag2num[child_tag]>mx:
            mx=tag2num[child_tag]
            mx_tag=child_tag

        tag+=child_tag*child_size
        size+=child_size
    id2size[pos]=size
    #id2tag[pos]=int(tag/size)
    id2tag[pos]=mx_tag
    return mx_tag,size,id2tag,id2size

def update(index, id2child, id2deep):  # 更新qu[index]的所有子节点的深度
    if len(id2child[index]) > 1:
        for node in id2child[index]:
            # qu[node]=(qu[node][0],qu[node][1],qu[node][2],qu[node][3],qu[node][4],qu[node][5],qu[node][6]+1)
            id2deep[node] = id2deep[node] + 1
            id2deep = update(node, id2child, id2deep)
    return id2deep


def get_deep(index, id2child, id2deep):
    deep = id2deep[index]
    if len(id2child[index]) > 1:
        for node in id2child[index]:
            deep = max(deep, get_deep(node, id2child, id2deep))
    return deep


def structual_entropy(edges, nodes, mx_deep,label,node_tags,dataset):
    nodes = np.array(nodes)
    edges = np.array(edges)
    id2index = {j: i+1 for i, j in enumerate(nodes)} #从1编号
    mapped_edge = np.array(list(map(id2index.get, edges.flatten())), dtype=np.int32).reshape(edges.shape)
    nodes = [id2index[id] for id in nodes]
    edges=list(mapped_edge)


    degree = {}
    g_dict = {}
    n, m, degree, g_dict, sum, edge = read_file(edges, degree, g_dict, nodes)
    #print("num of nodes:",n)
    h1 = 0
    #print(edges,nodes)
    for i in range(1, n + 1):
        h1 += (-degree[i] / (2.0 * sum) * math.log(degree[i] / (2.0 * sum), 2))
    #print(h1)
    nums = []
    for i in range(1, n + 1):
        nums.append(i)
    qu = [(0, 2 * sum, [], [], 0)]
    id2sister = {}
    id2child = {0: nums}
    id2deep = {0: 1}
    id2fa = {0: -1}
    I = {0:0}
    for i in range(1, n + 1):
        qu.append((degree[i], degree[i]))  # 分别表示团的割边数，度的和
        id2sister[i] = edge[i]
        id2child[i] = [i]
        id2deep[i] = 2
        id2fa[i] = 0
        I[i] = degree[i]
        I[0]+=degree[i]
    result = 0
    cnt = n + 1
    flag = True
    #flag=False
    flag2 = True
    #flag2=False
    delete_id = []
    # print(id2sister)
    iter = 1
    while (flag or flag2):
        # while(flag2):
        flag2 = True
        # while(flag2):
        if flag2:
            iter += 1
            #print(iter)
            mn = 1e9
            mx = 1e-6
            flag2 = False
            for i in range(1, len(qu)):
                if i in delete_id:
                    continue
                item1 = qu[i]
                g1 = item1[0]
                for j in id2sister[i]:

                    item2 = qu[j]
                    if len(id2child[id2fa[i]]) <= 2 or j in delete_id:
                        # print("error")
                        continue
                    g2 = item2[0]
                    # new_edge=item1[3]+item2[3]
                    v = item1[1] + item2[1]
                    # new_node=item1[2]+item2[2]
                    v_fa = qu[id2fa[i]][1]
                    if (i, j) in g_dict.keys():
                        g = g1 + g2 - 2 * g_dict[(i, j)]
                    else:
                        g = g1 + g2
                    # 按照combine后熵减小最多的两个团combine
                    # 深度不能超过max_deep
                    if (g1 + g2 - g) / (2 * sum) * math.log((v_fa) / v, 2) > mx and get_deep(i, id2child,
                                                                                             id2deep) + 1 <= mx_deep and get_deep(
                            j, id2child, id2deep) + 1 <= mx_deep:
                        mx = (g1 + g2 - g) / (2 * sum) * math.log((v_fa) / v, 2)
                        add = mx

                        ans = (g, v)
                        id1 = i
                        id2 = j
                        flag2 = True
            if flag2:
                # print(len(qu),index1,index2)
                #print('combine', id1, id2, cnt)
                # 更新父节点
                id2fa[cnt] = id2fa[id1]
                id2fa[id1] = cnt
                id2fa[id2] = cnt
                # 更新子节点
                id2child[cnt] = [id1, id2]
                fa_id = id2fa[cnt]
                # print('combine',fa_id,id1,id2)
                id2child[fa_id].remove(id1)
                id2child[fa_id].remove(id2)
                id2child[fa_id].append(cnt)
                # print(id2child)
                # 更新深度
                # print(qu[index1][0],qu[index2][0],ans[0])
                id2deep[cnt] = id2deep[id1]
                id2deep[id1] = id2deep[cnt] + 1
                id2deep[id2] = id2deep[cnt] + 1
                id2deep = update(id1, id2child, id2deep)
                id2deep = update(id2, id2child, id2deep)
                # print(mn)
                result += add
                # print(result)
                # 更新g_dict

                for i in range(0, len(qu)):
                    if id2deep[cnt] == id2deep[i] and id2fa[cnt] == id2fa[i] and i not in delete_id:
                        if (id1, i) in g_dict.keys():
                            c1 = g_dict[(id1, i)]
                        else:
                            c1 = 0
                        if (id2, i) in g_dict.keys():
                            c2 = g_dict[(id2, i)]
                        else:
                            c2 = 0
                        c = c1 + c2
                        if c > 0:
                            g_dict[(cnt, i)] = g_dict[(i, cnt)] = c

                # 更新id2sister:
                id2sister[id2].remove(id1)
                id2sister[id1].remove(id2)
                id2sister[cnt] = list(set(id2sister[id1] + id2sister[id2]))

                for id in id2sister[id1]:
                    id2sister[id].remove(id1)
                    id2sister[id].append(cnt)
                for id in id2sister[id2]:
                    id2sister[id].remove(id2)
                    if cnt not in id2sister[id]:
                        id2sister[id].append(cnt)
                id2sister[id1] = [id2]
                id2sister[id2] = [id1]
                # print(id1,id2sister[id1])
                # print(id2,id2sister[id2])
                # print(cnt,id2sister[cnt])
                # print(id2sister)
                '''
                for i in id2sister[cnt]:
                    if (id1, i) in g_dict.keys():
                        c1 = g_dict[(id1, i)]
                    else:
                        c1 = 0
                    if (id2, i) in g_dict.keys():
                        c2 = g_dict[(id2, i)]
                    else:
                        c2 = 0
                    c = c1 + c2
                    if c > 0:
                        g_dict[(cnt, i)] = g_dict[(i, cnt)] = c
                '''
                # 更新I
                qu.append(ans)
                I[cnt] = qu[id1][0] + qu[id2][0]
                I[id2fa[cnt]] = I[id2fa[cnt]] - (qu[id1][0] + qu[id2][0] - qu[cnt][0])
                #print(I)
                cnt += 1

        flag = True
        while (flag):
            iter += 1
            #print(iter)
            flag = False
            mx = 1e-5
            item1 = qu[cnt - 1]
            if len(id2child[id2fa[cnt - 1]]) <= 2:
                break
            v1 = item1[1]
            g1 = item1[0]
            for j in id2sister[cnt - 1]:
                # 计算merge cnt和j的收益

                item2 = qu[j]
                if j in delete_id:
                    continue
                v2 = item2[1]
                g2 = item2[0]
                # print(item1[2],item2[2],new_node)

                v12 = item1[1] + item2[1]

                if (cnt - 1, j) in g_dict.keys():
                    g12 = g1 + g2 - 2 * g_dict[(cnt - 1, j)]
                else:
                    g12 = g1 + g2

                v = item1[1] + item2[1]
                # new_node=item1[2]+item2[2]

                v_fa = qu[id2fa[cnt - 1]][1]

                I1 = I[cnt - 1] - g1
                I2 = I[j] - g2
                # print(I1, I2)
                # dif = (g1+g2-g12)/(2*sum)*math.log(v_fa/v,2) - (I1-g1)/(2*sum)*math.log(v/v1,2) - (I2 - g2)/(2*sum)*math.log(v/v2,2)
                dif = (g1 + g2 - g12) / (2 * sum) * math.log(v_fa, 2) + (I1) / (2 * sum) * math.log(v1, 2) \
                      + (I2) / (2 * sum) * math.log(v2, 2) - (I[cnt - 1] + I[j] - g12) / (2 * sum) * math.log(v, 2)
                # new_node=item1[2]+item2[2]
                # 计算merge后的熵
                '''
                after_merge = -g12 / (2 * sum) * math.log(v12 / v_fa, 2)
                for node in id2child[cnt - 1] + id2child[j]:
                    after_merge += -qu[node][0] / (2 * sum) * math.log(qu[node][1] / v12, 2)
                # print(after_merge)
                before_merge = -g1 / (2 * sum) * math.log(v1 / v_fa, 2) - g2 / (2 * sum) * math.log(v2 / v_fa, 2)
                for node in id2child[cnt - 1]:
                    before_merge += -qu[node][0] / (2 * sum) * math.log(qu[node][1] / v1, 2)
                for node in id2child[j]:
                    before_merge += -qu[node][0] / (2 * sum) * math.log(qu[node][1] / v2, 2)
                dif = before_merge - after_merge
                '''
                '''
                print(dif, dif2)
                if math.fabs(dif-dif2)>1e-3:
                    print("!!!!!!!!!!!!!!!!!!!!!")
                '''
                # print(before_merge,after_merge)

                if dif > mx:
                    mx = dif
                    ans = (g12, v12)
                    add = dif
                    id2 = j
                    flag = True
            if flag:
                id1 = cnt - 1
                if len(id2child[id1]) > 1:
                    delete_id.append(id1)
                if len(id2child[id2]) > 1:
                    delete_id.append(id2)

                #print('merge', id1, id2, cnt)
                # 更新父节点
                id2fa[cnt] = id2fa[id1]

                # 更新父亲id的子节点
                id2child[cnt] = id2child[id1] + id2child[id2]
                fa_id = id2fa[cnt]
                # print('merge',fa_id,id1,id2)
                id2child[fa_id].remove(id1)
                id2child[fa_id].remove(id2)
                id2child[fa_id].append(cnt)
                # print(id2child)
                # 更新深度和子节点的父节点
                id2deep[cnt] = id2deep[id1]
                for node in id2child[cnt]:
                    id2deep[node] = id2deep[cnt] + 1
                    id2fa[node] = cnt
                result += add
                '''
                for i in range(0, len(qu)):
                    if id2deep[cnt] == id2deep[i] and id2fa[cnt] == id2fa[i] and i not in delete_id:
                        if (id1, i) in g_dict.keys():
                            c1 = g_dict[(id1, i)]
                        else:
                            c1 = 0
                        if (id2, i) in g_dict.keys():
                            c2 = g_dict[(id2, i)]
                        else:
                            c2 = 0
                        c = c1 + c2
                        if c > 0:
                            g_dict[(cnt, i)] = g_dict[(i, cnt)] = c
                '''
                # 更新id2sister
                id2sister[id2].remove(id1)
                id2sister[id1].remove(id2)
                id2sister[cnt] = list(set(id2sister[id1] + id2sister[id2]))
                # print(cnt,id2sister[cnt],id2sister[id1],id2sister[id2])
                for id in id2sister[id1]:
                    id2sister[id].remove(id1)
                    id2sister[id].append(cnt)
                for id in id2sister[id2]:
                    id2sister[id].remove(id2)
                    if cnt not in id2sister[id]:
                        id2sister[id].append(cnt)
                for sub_id1 in id2child[id1] + id2child[id2]:
                    id2sister[sub_id1] = []
                    for sub_id2 in id2child[id1] + id2child[id2]:
                        if sub_id1 != sub_id2 and (sub_id1, sub_id2) in g_dict.keys():
                            id2sister[sub_id1].append(sub_id2)


                for i in id2sister[cnt]:
                    if (id1, i) in g_dict.keys():
                        c1 = g_dict[(id1, i)]
                    else:
                        c1 = 0
                    if (id2, i) in g_dict.keys():
                        c2 = g_dict[(id2, i)]
                    else:
                        c2 = 0
                    c = c1 + c2
                    if c > 0:
                        g_dict[(cnt, i)] = g_dict[(i, cnt)] = c

                # 更新I

                qu.append(ans)
                I[cnt] = I[id1] + I[id2]
                I[id2fa[cnt]] = I[id2fa[cnt]] - (qu[id1][0] + qu[id2][0] - qu[cnt][0])
                #print(I)
                cnt += 1

        flag = True

        while (flag):
            iter += 1
            #print(iter)
            flag = False
            mx = 1e-5
            item1 = qu[cnt - 1]
            if len(id2child[id2fa[cnt - 1]]) <= 2:
                break
            v1 = item1[1]
            g1 = item1[0]
            for j in id2sister[cnt - 1]:
                # 计算merge cnt和j的收益

                item2 = qu[j]
                if j in delete_id:
                    continue
                v2 = item2[1]
                g2 = item2[0]
                # print(item1[2],item2[2],new_node)
                v12 = item1[1] + item2[1]
                v = item1[1] + item2[1]

                if (cnt - 1, j) in g_dict.keys():
                    g12 = g1 + g2 - 2 * g_dict[(cnt - 1, j)]
                else:
                    g12 = g1 + g2
                v_fa = qu[id2fa[cnt - 1]][1]
                I1 = I[cnt - 1] - g1
                I2 = I[j] - g2
                dif = (g1+g2-g12)/(2*sum)*math.log(v_fa/v,2) - (I1)/(2*sum)*math.log(v/v1,2) - (I2)/(2*sum)*math.log(v/v2,2)
                #dif = (g1 + g2 - g12) / (2 * sum) * math.log(v_fa, 2) + (I1) / (2 * sum) * math.log(v1, 2) \
                      #+ (I2) / (2 * sum) * math.log(v2, 2) - (I[cnt - 1] + I[j] - g12) / (2 * sum) * math.log(v, 2)
                # new_node=item1[2]+item2[2]
                # 计算merge后的熵
                '''
                after_merge = -g12 / (2 * sum) * math.log(v12 / v_fa, 2)
                for node in id2child[cnt - 1] + id2child[j]:
                    after_merge += -qu[node][0] / (2 * sum) * math.log(qu[node][1] / v12, 2)
                # print(after_merge)
                before_merge = -g1 / (2 * sum) * math.log(v1 / v_fa, 2) - g2 / (2 * sum) * math.log(v2 / v_fa, 2)
                for node in id2child[cnt - 1]:
                    before_merge += -qu[node][0] / (2 * sum) * math.log(qu[node][1] / v1, 2)
                for node in id2child[j]:
                    before_merge += -qu[node][0] / (2 * sum) * math.log(qu[node][1] / v2, 2)
                dif2 = before_merge - after_merge
                '''
                #print("dif:",dif,dif2)
                # print(before_merge,after_merge)


                if dif >= mx:
                    mx = dif
                    ans = (g12, v12)
                    add = dif
                    id2 = j
                    flag = True
            if flag:
                id1 = cnt - 1
                if len(id2child[id1]) > 1:
                    delete_id.append(id1)
                if len(id2child[id2]) > 1:
                    delete_id.append(id2)

                #print('merge', id1, id2, cnt)
                # 更新父节点
                id2fa[cnt] = id2fa[id1]

                # 更新父亲id的子节点
                id2child[cnt] = id2child[id1] + id2child[id2]
                fa_id = id2fa[cnt]
                # print('merge',fa_id,id1,id2)
                id2child[fa_id].remove(id1)
                id2child[fa_id].remove(id2)
                id2child[fa_id].append(cnt)
                # print(id2child)
                # 更新深度和子节点的父节点
                id2deep[cnt] = id2deep[id1]
                for node in id2child[cnt]:
                    id2deep[node] = id2deep[cnt] + 1
                    id2fa[node] = cnt
                result += add
                '''
                for i in range(0, len(qu)):
                    if id2deep[cnt] == id2deep[i] and id2fa[cnt] == id2fa[i] and i not in delete_id:
                        if (id1, i) in g_dict.keys():
                            c1 = g_dict[(id1, i)]
                        else:
                            c1 = 0
                        if (id2, i) in g_dict.keys():
                            c2 = g_dict[(id2, i)]
                        else:
                            c2 = 0
                        c = c1 + c2
                        if c > 0:
                            g_dict[(cnt, i)] = g_dict[(i, cnt)] = c
                '''
                # 更新id2sister
                id2sister[id2].remove(id1)
                id2sister[id1].remove(id2)
                id2sister[cnt] = list(set(id2sister[id1] + id2sister[id2]))
                # print(cnt,id2sister[cnt],id2sister[id1],id2sister[id2])
                for id in id2sister[id1]:
                    id2sister[id].remove(id1)
                    id2sister[id].append(cnt)
                for id in id2sister[id2]:
                    id2sister[id].remove(id2)
                    if cnt not in id2sister[id]:
                        id2sister[id].append(cnt)
                for sub_id1 in id2child[id1] + id2child[id2]:
                    id2sister[sub_id1] = []
                    for sub_id2 in id2child[id1] + id2child[id2]:
                        if sub_id1 != sub_id2 and (sub_id1, sub_id2) in g_dict.keys():
                            id2sister[sub_id1].append(sub_id2)

                for i in id2sister[cnt]:
                    if (id1, i) in g_dict.keys():
                        c1 = g_dict[(id1, i)]
                    else:
                        c1 = 0
                    if (id2, i) in g_dict.keys():
                        c2 = g_dict[(id2, i)]
                    else:
                        c2 = 0
                    c = c1 + c2
                    if c > 0:
                        g_dict[(cnt, i)] = g_dict[(i, cnt)] = c


                qu.append(ans)
                I[cnt] = I[id1] + I[id2]
                I[id2fa[cnt]] = I[id2fa[cnt]] - (qu[id1][0] + qu[id2][0]-qu[cnt][0])
                cnt += 1
    g = nx.Graph()
    ids = []
    edges = []
    id2tag = {}
    id2size = {}
    id2adj=[]
    # 输出树上每个节点的信息
    for i, item in enumerate(qu):
        if i not in delete_id:
            #print(i, id2fa[i], id2deep[i], id2child[i])
            ids.append(i)
            if len(id2child)>1:
                if i==0:
                    tem=id2child[i]
                    #tem=[]
                else:
                    tem=[id2fa[i]]+id2child[i]
                    #tem = [id2fa[i]]
            else:
                tem = [id2fa[i]]
            id2adj.append(tem)
            for child in id2child[i]:
                #edges.append((i, child))
                edges.append((child, i))


    for i,tag in enumerate(node_tags):
        id2tag[i+1]=tag
        id2size[i+1]=1
    _,_,id2tag,id2size=dfs(0,id2tag,id2size,id2child)
    sort_tag=sorted(id2tag.items(), key=lambda x: x[0])
    new_tag=[item[1] for item in sort_tag]


    ids=np.array(ids)
    edges=np.array(edges)

    id2index = {j: i for i, j in enumerate(ids)}
    for i,item in enumerate(id2adj):
        id2adj[i]=[id2index[adj] for adj in id2adj[i]]
    mapped_edge = np.array(list(map(id2index.get, edges.flatten())), dtype=np.int32).reshape(edges.shape)
    ids=[id2index[id] for id in ids]
    g.add_nodes_from(list(ids))
    g.add_edges_from(list(mapped_edge))
    g.label=label
    g.node_tags=new_tag

    '''
    if dataset!='':
        with open('../../data/' + dataset + '/' + dataset + '_aug_3layer.txt', 'a', encoding='utf8') as f1:
            f1.write(str(len(ids)) +' '+str(label)+ '\n')
            for i,adj in enumerate(id2adj):
                num_adj=len(adj)
                adj=[str(item) for item in adj]
                adj_str=' '.join(adj)
                f1.write(str(new_tag[i]) + ' ' + str(num_adj) +' '+adj_str+  '\n')
    '''
    #print(h1, h1-result)
    return g,result

def graph_augment(g,dataset):
    #g有node,edge,node_tags,g.label
    edges=list(g.edges())
    nodes=list(g.nodes())
    max_deep=4
    label=g.label
    return structual_entropy(edges, nodes, max_deep,label,g.node_tags,dataset)







