import copy
from tqdm import tqdm
import torch
import torch.nn as nn

from model.utils import create_optimizer, accuracy

import sys

import importlib
from pygod.utils import load_data
from pygod.metrics import eval_roc_auc,eval_average_precision,eval_ndcg,eval_precision_at_k,eval_recall_at_k
from pygod.models import ADANET
# from torch_geometric.utils import to_dense_adj  # 移除稠密操作相关导入
import numpy as np

anomaly_num_dict={'weibo':868,'reddit':366,'disney':6,'books':28,'enron':5,'inj_cora':138,'inj_amazon':694,'inj_flickr':4414}

def god_evaluation(data_name,model_name,attr_encoder_name,struct_encoder_name,topology_encoder_name,
attr_decoder_name,struct_decoder_name,attr_ssl_model,struct_ssl_model,topology_ssl_model,graph, x, 
aggr_f,lr_f, max_epoch_f, alpha_f,dropout_f,loss_f,loss_weight_f,T_f,num_hidden,node_encoder_num_layers,edge_encoder_num_layers,subgraph_encoder_num_layers,
attr_decoder_num_layers=1,struct_decoder_num_layers=1,use_ssl=False,use_encoder_num=1,attention=None,sparse_attention_weight=0.001,
theta=1.001,eta=1.001):
    
    if use_encoder_num==1:
        attr_ssl_model.eval()
    if use_encoder_num==2:
        attr_ssl_model.eval()
        struct_ssl_model.eval()
    if use_encoder_num==3:
        attr_ssl_model.eval()
        struct_ssl_model.eval()
        topology_ssl_model.eval()
        
    
    model= eval(model_name)(epoch=max_epoch_f,aggr=aggr_f,hid_dim=num_hidden,alpha=alpha_f,dropout=dropout_f,\
        lr=lr_f,loss_name=loss_f,loss_weight=loss_weight_f,T=T_f,use_encoder_num=use_encoder_num,attention=attention,\
            attr_encoder_name=attr_encoder_name,struct_encoder_name=struct_encoder_name,topology_encoder_name=topology_encoder_name,attr_decoder_name=attr_decoder_name,struct_decoder_name=struct_decoder_name,\
            node_encoder_num_layers=node_encoder_num_layers,edge_encoder_num_layers=edge_encoder_num_layers,subgraph_encoder_num_layers=subgraph_encoder_num_layers,\
                attr_decoder_num_layers=attr_decoder_num_layers,\
                struct_decoder_num_layers=struct_decoder_num_layers,sparse_attention_weight=sparse_attention_weight,theta=theta,eta=eta)

    if use_ssl and use_encoder_num>0:
        if use_encoder_num==1:
            model.fit(graph,pretrain_attr_encoder=attr_ssl_model.encoder,pretrain_struct_encoder=None,pretrain_topology_encoder=None)
        elif use_encoder_num==2:
            model.fit(graph,pretrain_attr_encoder=attr_ssl_model.encoder,pretrain_struct_encoder=struct_ssl_model.encoder,pretrain_topology_encoder=None) 
        elif use_encoder_num==3: 
            model.fit(graph,pretrain_attr_encoder=attr_ssl_model.encoder,pretrain_struct_encoder=struct_ssl_model.encoder,pretrain_topology_encoder=topology_ssl_model.encoder)       
        else:
            assert(f'wrong encoder num: {use_encoder_num}')
    else:
        model.fit(graph)
    labels = model.predict(graph)

    outlier_scores= model.decision_function(graph)
    edge_outlier_scores=model.decision_struct_function(graph)

    auc_score = eval_roc_auc(graph.y.bool().cpu().numpy(), outlier_scores)
    ap_score = eval_average_precision(graph.y.bool().cpu().numpy(), outlier_scores)
    # 检查异常点数量
    labels_np = graph.y.bool().cpu().numpy()
    num_anomaly = int(labels_np.sum())
    print(f"异常点数量: {num_anomaly}")
    # 检查分数方向
    outlier_scores_np = np.array(outlier_scores)
    topk = min(10, len(outlier_scores_np))
    topk_idx = np.argsort(outlier_scores_np)[-topk:][::-1]
    print(f"前{topk}分数: {outlier_scores_np[topk_idx]}")
    print(f"前{topk}分数对应标签: {labels_np[topk_idx]}")
    # Rec@K自适应
    k = min(10, num_anomaly) if num_anomaly > 0 else 1
    rec_at_k = eval_recall_at_k(labels_np, outlier_scores_np, k=k)
    print(f"实际用于Rec@K的K值: {k}")
    # 自动检测分数方向
    rec_at_k_neg = eval_recall_at_k(labels_np, -outlier_scores_np, k=k)
    if rec_at_k == 0 and rec_at_k_neg > rec_at_k:
        print("检测到分数方向可能反了，自动采用负分数方向！")
        outlier_scores_np = -outlier_scores_np
        rec_at_k = rec_at_k_neg

    # 异常点分数分布分析
    anomaly_scores = outlier_scores_np[labels_np == 1]
    print("异常点分数分布（前20个）:", anomaly_scores[:20])
    print("异常点分数最大值:", anomaly_scores.max() if len(anomaly_scores) > 0 else 'N/A')
    print("所有分数最大值:", outlier_scores_np.max())
    # 随机分数Rec@K
    random_scores = np.random.rand(len(labels_np))
    rec_at_k_random = eval_recall_at_k(labels_np, random_scores, k=k)
    print(f"随机分数Rec@{k}: {rec_at_k_random}")

    print(f'auc_score: {auc_score:.4f}',)
    rec_at_k = float(rec_at_k)
    print(f"最终写入csv的Rec@K: {rec_at_k}, K_used: {k}")

    return auc_score, ap_score, None, rec_at_k, k, outlier_scores_np, edge_outlier_scores
