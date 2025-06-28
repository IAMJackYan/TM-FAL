from torch.utils.data import DataLoader
from data.sampler import SubsetSequentialSampler
import random
import torch
import pdb
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import torch.distributions as dist
import logging
import os
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import copy

def fl_duc(global_model, local_model, dataloader, client_idx=0, round_idx=0):
    global_model.eval()
    local_model.eval()

    g_u_data_list = torch.tensor([]).cuda()
    l_u_data_list = torch.tensor([]).cuda()
    g_u_dis_list = torch.tensor([]).cuda()

    l_feature_list = torch.tensor([]).cuda()

    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image = data['image']
            image = image.cuda()

            # global model
            g_logit, _, _, _ = global_model(image)
            alpha = F.relu(g_logit) + 1 # global
            total_alpha = torch.sum(alpha, dim=1, keepdim=True)
            dirichlet = dist.Dirichlet(alpha)
            g_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
            g_u_dis = dirichlet.entropy()

            # local model
            l_logit, _, _, block_features = local_model(image)
            l_feature = F.adaptive_avg_pool2d(block_features[-1], 3).flatten(start_dim=1)
            l_feature_list = torch.cat((l_feature_list, l_feature))
            alpha = F.relu(l_logit) + 1 # local
            total_alpha = torch.sum(alpha, dim=1, keepdim=True)
            l_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)

            g_u_data_list = torch.cat((g_u_data_list, g_u_data))
            l_u_data_list = torch.cat((l_u_data_list, l_u_data))
            g_u_dis_list = torch.cat((g_u_dis_list, g_u_dis))

    return g_u_data_list, l_u_data_list, g_u_dis_list, l_feature_list

    
def relaxation(u_rank_arg, l_feature_list, neighbor_num, query_num, unlabeled_len, cosine=0.85):
    query_flag = torch.zeros(unlabeled_len).cuda()

    chosen_idx = []
    ignore_cnt = 0
    for i in u_rank_arg: 
        if len(chosen_idx) == query_num:
            break

        cos_sim = pairwise_cosine_similarity(l_feature_list[i:i+1,:], l_feature_list)[0]
        neighbor_arg = torch.argsort(-cos_sim)    # descending order
        neighbor_arg = neighbor_arg[cos_sim[neighbor_arg]>cosine][1:1+neighbor_num]

        neighbor_flag = query_flag[neighbor_arg]
        if neighbor_flag.sum() == 0 or len(neighbor_arg)<neighbor_num:   
            query_flag[i] = 1
            chosen_idx.append(i.item())
        else:
            ignore_cnt += 1
            continue

    # logging.info('ignore num: {}'.format(ignore_cnt))

    remain_idx = list(set(range(unlabeled_len))- set(chosen_idx))
    rank_arg = remain_idx + chosen_idx

    return rank_arg


def loss_rank(global_model, dataloader):
    global_model.eval()
    loss_all = torch.tensor([]).cuda()
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            
            image, label = data['image'], data['label']
            image = image.cuda()
            label = label.cuda()
            logit = global_model(image)[0]    # logit, pred, embedding, block_output
            loss = criterion(logit, label)
            loss_all = torch.cat((loss_all, loss))

        
        rank = torch.argsort(loss_all)
        return rank.cpu().numpy()

            
def al_timeu(global_model, dataloader, model_pools, decision_model):
    global_model.eval()
    decision_model.eval()
    g_u_data_list = torch.tensor([]).cuda()
    data_label = torch.tensor([]).cuda()

    mean_pred_list = torch.tensor([]).cuda()
    deci_pred_list = torch.tensor([]).cuda()
    label_list = torch.tensor([]).cuda()

    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image = data['image']
            label = data['label']
            image = image.cuda()
            label = label.cuda()
            logits_list = []
            outputs_list = []
            for param in model_pools:
                global_model.load_state_dict(param)
                _, logits, outputs, _ = global_model(image)
                logits_list.append(logits)
                outputs_list.append(outputs)
            logits_tensor = torch.stack(logits_list, dim=1)
            mean_logits = logits_tensor.mean(dim=1)
            mean_label = mean_logits.argmax(dim=1)
            prob, _ = torch.max(mean_logits, dim=1)

            mean_pred_list = torch.cat((mean_pred_list, torch.argmax(mean_logits, dim=1)))
            label_list = torch.cat((label_list, label))

            outputs_tensor = torch.stack(outputs_list, dim=1)
            u = torch.std(outputs_tensor, dim=1)
            u = u.mean(dim=1)
            u = u/(prob + 1e-14)

            # 
            _, deci_logits, _, _ = decision_model(image)
            deci_label = deci_logits.argmax(dim=1)
            deci_pred_list = torch.cat((deci_pred_list, deci_label))           

            g_u_data_list = torch.cat((g_u_data_list, u))

            data_label = torch.cat((data_label, deci_label))
        
        data_label = data_label.long()
    return g_u_data_list, data_label

            


def query_samples(al_method, global_model, local_model, data_unlabeled, unlabeled_set, labeled_set, query_num, num_per_class, client_idx, round_idx, args, model_pools=None, save_dir = None, zero_model=None, cluster_size=10, s_step=1):
    unlabeled_len = len(unlabeled_set)
    query_model = args.query_model
    dataset = args.dataset

    if al_method == 'Random':
        rank_arg = list(range(unlabeled_len))
        random.shuffle(rank_arg)
    
    elif al_method == 'FEAL':
        unlabeled_loader = DataLoader(dataset=data_unlabeled,
                                        batch_size=args.batch_size,
                                        sampler = SubsetSequentialSampler(unlabeled_set), 
                                        num_workers=1,
                                        pin_memory=True)
        g_data_list, l_data_list, u_dis_list, l_feature_list = fl_duc(global_model, local_model, unlabeled_loader, client_idx, round_idx)

        # Stage 1: uncertainty calibration
        u_dis_norm = (u_dis_list-u_dis_list.min()) / (u_dis_list.max()-u_dis_list.min())
        uncertainty = u_dis_norm * (g_data_list+l_data_list)
        u_rank_arg = torch.argsort(-uncertainty).cpu().numpy()    # descending order

        # Stage 2: relaxation
        rank_arg = relaxation(u_rank_arg=u_rank_arg, l_feature_list=l_feature_list, neighbor_num=args.n_neighbor, query_num=query_num, unlabeled_len=unlabeled_len, cosine=args.cosine)
    
    elif al_method == 'ours':
        
        unlabeled_loader = DataLoader(dataset=data_unlabeled,
                                        batch_size=args.batch_size,
                                        sampler = SubsetSequentialSampler(unlabeled_set), 
                                        num_workers=1,
                                        pin_memory=True)
        g_data_list, labels = al_timeu(global_model, unlabeled_loader, model_pools=model_pools, decision_model = copy.deepcopy(global_model))
        
        rank = torch.argsort(-g_data_list)
        rank_list = rank.cpu().numpy()

        sort_labels = labels[rank].cpu().numpy()
        class_labels = np.unique(sort_labels)
        result = []

        label_indexs = {}

        max_length = 0
        for i in class_labels:
            label_indexs[i] = np.where(sort_labels == i)[0]
            max_length = max(max_length, len(label_indexs[i]))
        
        for i in range(max_length):
            for j in class_labels:
                if len(label_indexs[j]) >i:
                    result.append(rank_list[label_indexs[j][i]])
        rank_arg = np.array(result)
        # rank_arg = rank_list
        rank_arg = np.flip(rank_arg)
        rank_arg = rank_arg.copy()

    return rank_arg