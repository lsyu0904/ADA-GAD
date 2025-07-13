
import logging
import numpy as np
from tqdm import tqdm
import torch

import sys
import os
# 只将ADA-GAD/pygod目录加入sys.path，确保导入正确的pygod
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PYGOD_DIR = os.path.join(CUR_DIR, 'pygod')
sys.path.insert(0, PYGOD_DIR)
import importlib

from pygod.utils.utility import load_data
from pygod.metrics import eval_roc_auc,eval_average_precision,eval_ndcg,eval_precision_at_k,eval_recall_at_k
from pygod.models import ADANET
# from torch_geometric.utils import to_dense_adj,add_remaining_self_loops,add_self_loops  # 移除稠密操作相关导入

from model.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from model.models import build_model
from model.god import god_evaluation

import logging
import pandas as pd

def save_results_to_csv(dataset, auc, ap, rec_at_k, k_used, csv_path='results.csv'):
    import os
    file_exists = os.path.isfile(csv_path)
    df = pd.DataFrame([{
        'dataset': dataset,
        'AUROC': auc,
        'AUPRC': ap,
        'Rec@K': rec_at_k,
        'K_used': k_used
    }])
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, model_name,aggr_f,lr_f, max_epoch_f, alpha_f,dropout_f,loss_f,loss_weight_f,T_f, num_hidden=16,logger=None,use_ssl=False,return_edge_score=False):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter =range(max_epoch_f)

    outlier_score_list=[]
    edge_outlier_score_list=[]
    for epoch in epoch_iter:
        model.train()
        loss, loss_dict,final_edge_mask_rate = model(x, graph.edge_index)
        if loss is None:
            print(f"[pretrain] loss为None，因节点数过大或to_dense_adj跳过，本epoch不训练。")
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

    # return best_model
    return model,np.array(outlier_score_list),np.array(edge_outlier_score_list)

def main(args):
    # 替换原有device赋值
    # device = args.device if args.device >= 0 else "cpu"
    device = torch.device(f"cuda:{args.device}" if hasattr(args, 'device') and args.device is not None and int(args.device) >= 0 and torch.cuda.is_available() else "cpu")
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch

    num_hidden = args.num_hidden

    node_encoder_num_layers = args.node_encoder_num_layers
    edge_encoder_num_layers = args.edge_encoder_num_layers
    subgraph_encoder_num_layers = args.subgraph_encoder_num_layers

    attr_decoder_num_layers= args.attr_decoder_num_layers
    struct_decoder_num_layers= args.struct_decoder_num_layers

    attr_encoder_name = args.attr_encoder
    struct_encoder_name = args.struct_encoder
    topology_encoder_name=args.topology_encoder

    attr_decoder_name = args.attr_decoder
    struct_decoder_name= args.struct_decoder

    replace_rate = args.replace_rate
    weight_decay=args.weight_decay
    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr

    model_name=args.model_name
    aggr_f=args.aggr_f
    max_epoch_f = args.max_epoch_f
    lr_f = args.lr_f
    alpha_f= args.alpha_f
    dropout_f=args.dropout_f
    loss_f=args.loss_f
    loss_weight_f=args.loss_weight_f
    T_f=args.T_f

    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    max_pu_epoch=args.max_pu_epoch
    each_pu_epoch=args.each_pu_epoch

    """
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        x : torch.Tensor
            Attribute (feature) of nodes.
        s : torch.Tensor
            Adjacency matrix of the graph.
        edge_index : torch.Tensor
            Edge list of the graph.
    """
    graph=load_data(dataset_name)
    graph.edge_index=add_remaining_self_loops(graph.edge_index)[0]
    num_features=graph.x.size()[1]
    num_classes=4

    args.num_features = num_features

    auc_score_list = []

    attr_mask,struct_mask=None,None
    pretrain_auc_score_list=[]
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        seed=int(seed)
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{attr_decoder_type}_{struct_decoder_type}")
        else:
            logger = None

        attr_model,struct_model,topology_model = build_model(args)
        attr_model.to(device)
        struct_model.to(device)
        topology_model.to(device)

        if args.use_ssl:
            attr_remask=None
            struct_remask=None
            print('======== train attr encoder ========')
            if args.use_encoder_num>=1:

                optimizer = create_optimizer(optim_type, attr_model, lr, weight_decay)

                if use_scheduler:
                    logging.info("Use schedular")
                    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                    # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                else:
                    scheduler = None
                    
                x = graph.x
                if not load_model:
                    attr_model,attr_outlier_list,_= pretrain(attr_model, graph, x, optimizer, max_epoch, device, scheduler, model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                    attr_model = attr_model.cpu()

                if load_model:
                    logging.info("Loading Model ... ")
                    attr_model.load_state_dict(torch.load("checkpoint.pt"))
                if save_model:
                    logging.info("Saveing Model ...")
                    torch.save(attr_model.state_dict(), "checkpoint.pt")
                
                attr_model = attr_model.to(device)
                attr_model.eval()

            print('======== train struct encoder ========')
            if args.use_encoder_num>=2:

                optimizer = create_optimizer(optim_type, struct_model, lr, weight_decay)

                if use_scheduler:
                    logging.info("Use schedular")
                    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                    # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                else:
                    scheduler = None
                    
                x = graph.x
                if not load_model:
                    struct_model,struct_node_outlier_list,struct_outlier_list= pretrain(struct_model, graph, x, optimizer, max_epoch, device, scheduler,  model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                    struct_model = struct_model.cpu()

                if load_model:
                    logging.info("Loading Model ... ")
                    struct_model.load_state_dict(torch.load("checkpoint.pt"))
                if save_model:
                    logging.info("Saveing Model ...")
                    torch.save(struct_model.state_dict(), "checkpoint.pt")
                
                struct_model = struct_model.to(device)
                struct_model.eval()

            print('======== train topology encoder ========')
            if args.use_encoder_num>=3:

                optimizer = create_optimizer(optim_type, topology_model, lr, weight_decay)

                if use_scheduler:
                    logging.info("Use schedular")
                    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                    # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                else:
                    scheduler = None
                    
                x = graph.x
                if not load_model:
                    topology_model,topology_node_outlier_list,topology_outlier_list= pretrain(topology_model, graph, x, optimizer, max_epoch, device, scheduler,  model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                    topology_model = topology_model.cpu()

                if load_model:
                    logging.info("Loading Model ... ")
                    topology_model.load_state_dict(torch.load("checkpoint.pt"))
                if save_model:
                    logging.info("Saveing Model ...")
                    torch.save(topology_model.state_dict(), "checkpoint.pt")
                
                topology_model = topology_model.to(device)
                struct_model.eval()

        print('finish one train!')
        auc_score, ap_score, ndcg_score, rec_at_k, k_used, outlier_scores, edge_outlier_scores = god_evaluation(
            dataset_name, model_name, attr_encoder_name, struct_encoder_name, topology_encoder_name,
            attr_decoder_name, struct_decoder_name, attr_model, struct_model, topology_model, graph, graph.x,
            aggr_f, lr_f, max_epoch, alpha_f, dropout_f, loss_f, loss_weight_f, T_f, args.num_hidden,
            node_encoder_num_layers, edge_encoder_num_layers, subgraph_encoder_num_layers,
            attr_decoder_num_layers, struct_decoder_num_layers, use_ssl=args.use_ssl, use_encoder_num=args.use_encoder_num,
            attention=args.attention, sparse_attention_weight=args.sparse_attention_weight, theta=args.theta, eta=args.eta
        )
        auc_score_list.append(auc_score)
        # 保存本次实验结果到csv，包含实际K值
        save_results_to_csv(dataset_name, auc_score, ap_score, rec_at_k, k_used)
        # 自动释放显存和内存，防止OOM
        import gc
        del attr_model, struct_model, topology_model, graph
        torch.cuda.empty_cache()
        gc.collect()

        if logger is not None:
            logger.finish()

    final_auc, final_auc_std = np.mean(auc_score_list), np.std(auc_score_list)

    # 假设ap_score, pk_score为最后一次god_evaluation的结果
    # 若有多次可自行调整为平均
    save_results_to_csv(dataset_name, final_auc, ap_score, pk_score, '')

    print(f"# final_auc: {final_auc*100:.2f}±{final_auc_std*100:.2f}")




# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    import torch
    print("CUDA可用：", torch.cuda.is_available())
    print("当前设备：", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
    print("GPU名称：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")
    args = build_args()
    print("step1: 参数解析完成", flush=True)
    if args.use_cfg:
        args = load_best_configs(args, "config_ada-gad.yml")

    if args.alpha_f=='None':
        args.alpha_f=None

    if args.all_encoder_layers!=0:
        args.node_encoder_num_layers=args.all_encoder_layers
        args.edge_encoder_num_layers=args.all_encoder_layers
        args.subgraph_encoder_num_layers=args.all_encoder_layers

    print(args)
    print("step2: 数据加载前", flush=True)
    # main函数内数据加载后也加print
    def main_with_print(args):
        # 确保模型和数据都用.to(device)
        # 例如：
        # attr_model.to(device)
        # struct_model.to(device)
        # topology_model.to(device)
        # graph = graph.to(device)
        # x = feat.to(device)
        # 强制使用GPU（cuda:0），如不可用则用cpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("强制使用device：", device)
        seeds = args.seeds
        dataset_name = args.dataset
        max_epoch = args.max_epoch

        num_hidden = args.num_hidden

        node_encoder_num_layers = args.node_encoder_num_layers
        edge_encoder_num_layers = args.edge_encoder_num_layers
        subgraph_encoder_num_layers = args.subgraph_encoder_num_layers

        attr_decoder_num_layers= args.attr_decoder_num_layers
        struct_decoder_num_layers= args.struct_decoder_num_layers

        attr_encoder_name = args.attr_encoder
        struct_encoder_name = args.struct_encoder
        topology_encoder_name=args.topology_encoder

        attr_decoder_name = args.attr_decoder
        struct_decoder_name= args.struct_decoder

        replace_rate = args.replace_rate
        weight_decay=args.weight_decay
        optim_type = args.optimizer 
        loss_fn = args.loss_fn

        lr = args.lr

        model_name=args.model_name
        aggr_f=args.aggr_f
        max_epoch_f = args.max_epoch_f
        lr_f = args.lr_f
        alpha_f= args.alpha_f
        dropout_f=args.dropout_f
        loss_f=args.loss_f
        loss_weight_f=args.loss_weight_f
        T_f=args.T_f

        load_model = args.load_model
        save_model = args.save_model
        logs = args.logging
        use_scheduler = args.scheduler

        max_pu_epoch=args.max_pu_epoch
        each_pu_epoch=args.each_pu_epoch

        print("step2: 数据加载开始", flush=True)
        graph=load_data(dataset_name)
        print("step2: 数据加载完成", flush=True)
        graph = graph.to(device)
        print("graph.x 设备：", graph.x.device)
        num_nodes = graph.x.size(0)
        # 移除graph.s = to_dense_adj(...)，全程只用edge_index
        # if num_nodes > 10000:
        #     print(f"节点数为{num_nodes}，过大，跳过to_dense_adj以避免显存溢出！")
        #     graph.s = None
        # else:
        #     graph.s = to_dense_adj(graph.edge_index)[0]
        num_features = graph.x.size()[1]
        num_classes = 4
        args.num_features = num_features
        auc_score_list = []
        attr_mask, struct_mask = None, None
        pretrain_auc_score_list = []
        for i, seed in enumerate(seeds):
            print(f"####### Run {i} for seed {seed}")
            seed = int(seed)
            set_random_seed(seed)

            if logs:
                logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{attr_decoder_type}_{struct_decoder_type}")
            else:
                logger = None

            print("step3: 模型初始化前", flush=True)
            attr_model, struct_model, topology_model = build_model(args)
            attr_model = attr_model.to(device)
            struct_model = struct_model.to(device)
            topology_model = topology_model.to(device)
            print("attr_model 设备：", next(attr_model.parameters()).device)
            print("struct_model 设备：", next(struct_model.parameters()).device)
            print("topology_model 设备：", next(topology_model.parameters()).device)
            print("step3: 模型初始化完成", flush=True)

            if args.use_ssl:
                attr_remask=None
                struct_remask=None
                print('======== train attr encoder ========')
                if args.use_encoder_num>=1:

                    optimizer = create_optimizer(optim_type, attr_model, lr, weight_decay)

                    if use_scheduler:
                        logging.info("Use schedular")
                        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                        # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                                # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                    else:
                        scheduler = None
                        
                    x = graph.x
                    if not load_model:
                        attr_model,attr_outlier_list,_= pretrain(attr_model, graph, x, optimizer, max_epoch, device, scheduler, model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                        attr_model = attr_model.cpu()

                    if load_model:
                        logging.info("Loading Model ... ")
                        attr_model.load_state_dict(torch.load("checkpoint.pt"))
                    if save_model:
                        logging.info("Saveing Model ...")
                        torch.save(attr_model.state_dict(), "checkpoint.pt")
                    
                    attr_model = attr_model.to(device)
                    attr_model.eval()

                print('======== train struct encoder ========')
                if args.use_encoder_num>=2:

                    optimizer = create_optimizer(optim_type, struct_model, lr, weight_decay)

                    if use_scheduler:
                        logging.info("Use schedular")
                        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                        # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                                # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                    else:
                        scheduler = None
                        
                    x = graph.x
                    if not load_model:
                        struct_model,struct_node_outlier_list,struct_outlier_list= pretrain(struct_model, graph, x, optimizer, max_epoch, device, scheduler,  model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                        struct_model = struct_model.cpu()

                    if load_model:
                        logging.info("Loading Model ... ")
                        struct_model.load_state_dict(torch.load("checkpoint.pt"))
                    if save_model:
                        logging.info("Saveing Model ...")
                        torch.save(struct_model.state_dict(), "checkpoint.pt")
                    
                    struct_model = struct_model.to(device)
                    struct_model.eval()

                print('======== train topology encoder ========')
                if args.use_encoder_num>=3:

                    optimizer = create_optimizer(optim_type, topology_model, lr, weight_decay)

                    if use_scheduler:
                        logging.info("Use schedular")
                        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                        # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                                # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                    else:
                        scheduler = None
                        
                    x = graph.x
                    if not load_model:
                        topology_model,topology_node_outlier_list,topology_outlier_list= pretrain(topology_model, graph, x, optimizer, max_epoch, device, scheduler,  model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                        topology_model = topology_model.cpu()

                    if load_model:
                        logging.info("Loading Model ... ")
                        topology_model.load_state_dict(torch.load("checkpoint.pt"))
                    if save_model:
                        logging.info("Saveing Model ...")
                        torch.save(topology_model.state_dict(), "checkpoint.pt")
                    
                    topology_model = topology_model.to(device)
                    struct_model.eval()

            print('finish one train!')
            auc_score,ap_score,ndcg_score,pk_score,rk_score,final_outlier,_= god_evaluation(dataset_name,model_name,attr_encoder_name,struct_encoder_name,topology_encoder_name,attr_decoder_name,struct_decoder_name,attr_model,struct_model,topology_model,graph, graph.x, aggr_f,lr_f, max_epoch, alpha_f,dropout_f,loss_f,loss_weight_f,T_f,args.num_hidden,node_encoder_num_layers,edge_encoder_num_layers,subgraph_encoder_num_layers,attr_decoder_num_layers,struct_decoder_num_layers,use_ssl=args.use_ssl,use_encoder_num=args.use_encoder_num,attention=args.attention,sparse_attention_weight=args.sparse_attention_weight,theta=args.theta,eta=args.eta)
            auc_score_list.append(auc_score)

            if logger is not None:
                logger.finish()

        final_auc, final_auc_std = np.mean(auc_score_list), np.std(auc_score_list)

        # 假设ap_score, pk_score为最后一次god_evaluation的结果
        # 若有多次可自行调整为平均
        save_results_to_csv(dataset_name, final_auc, ap_score, pk_score, '')

        print(f"# final_auc: {final_auc*100:.2f}±{final_auc_std*100:.2f}")
    main_with_print(args)
    

