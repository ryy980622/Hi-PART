# Hi-PART
This is our Pytorch implementation for the paper:

>Yuyang Ren, Haonan Zhang, Luoyi Fu, Xinbing Wang, Xinde Cao, Fei Long and Chenghu Zhou. Hi-PART: Going beyond Graph Pooling with Hierarchical Partition Tree for Graph-level Representation Learning

Author: Yuyang Ren (renyuyang@sjtu.edu.cn)

## Introduction
Hi-PART is a simple yet effective graph neural network (GNN) framework with hierarchical Partition Tree (HPT) for graph-level representation learning. By employing GNNs to summarize node features into the graph feature based on HPT's hierarchical structure, Hi-PART is able to adequately leverage the graph structure information and provably goes beyond the power of the WL test.


## Requirement
The code has been tested running under Python 3.7.0. The required packages are as follows:
* torch == 1.10.1
* numpy == 1.19.5
* networkx == 2.5.1


* Preprocess

```
python src/utils/graph_augmentation.py 
```

* Train

```
python src/main.py 
```

```

Some important hyperparameters:
* `lr`
  * It indicates the learning rates. 
  * The learning rate is searched in {1e-5, 1e-4, 3e-4,1e-3}.

* `batch`
  * It indicates the batch size. 
  * We search the batch size within {16, 32}.

* `h_dim'`
  * It indicates the latent dimension of node embeddings. 
  * We search the latent dimension within {16, 32, 128}.

* `K'`
  * It indicates the height of HPT. 
  * We search the height of HPT within {3, 4, 5}.


* `L_o'`
  * It indicates the GNN layer number of original graphs. 
  * We search L_o within {1, 2, 3}.


* `L_t'`
  * It indicates the GNN layer number of HPTs. 
  * We search L_o within {2, 3, 4}.

## Dataset
We provide one processed dataset: MUTAG.
* `MUTAG.txt`
  * Train file.
  * For each graph, the first line denotes the graph size and label.
  * Then each line denotes the edge set of each node.

* `MUTAG_aug.txt`
  * The preprocessed HPT file.
  * The format of this file is the same with train file.


  
