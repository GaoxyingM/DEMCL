#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import json
import argparse
import time
from tqdm import tqdm
from itertools import product
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import torch
import torch.optim as optim
from utility import Datasets
from models.DSCBR import DSCBR
from models.Diffusion import GaussianDiffusion, Denoise

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import TruncatedSVD

def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="Youshu", type=str, help="which dataset to use, options: NetEase, Youshu, iFashion")
    parser.add_argument("-m", "--model", default="DSCBR", type=str, help="which model to use, options: DSCBR")
    parser.add_argument("-i", "--info", default="", type=str, help="any auxilary info that will be appended to the log file name")

    # --begin add diffusion
    parser.add_argument('-steps', type=int ,default=5)
    parser.add_argument('--d_emb_size', type=int, default=10) #?
    parser.add_argument('--norm', type=bool, default=False)
    parser.add_argument('-dims', type=str, default='[1000]')
    parser.add_argument('-noise_scale', type=float, default=0.1)
    parser.add_argument('-noise_min', type=float, default=0.0001)
    parser.add_argument('-noise_max', type=float, default=0.02)
    parser.add_argument('-sampling_noise', type=bool, default=False)
    parser.add_argument('-sampling_steps', type=int,default=0)

    parser.add_argument('-keepRate', default=0.5, type=float, help='ratio of edges to keep')

    parser.add_argument('-e_loss', type=float, default=0.1)
    parser.add_argument('-rebuild_k', type=int, default=1)
    parser.add_argument('-ris_adj_lambda', type=float, default=0.2)
    parser.add_argument('-ris_lambda', type=float, default=0.5)

    parser.add_argument('-gnn_layer', default=1, type=int, help='number of gnn layers')
    # -- end add diffusion

    args = parser.parse_args()

    return args


def main():
    conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")

    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]

    assert paras["model"] in ["DSCBR"], "Pls select models from: DSCBR"

    if "_" in dataset_name:
        conf = conf[dataset_name.split("_")[0]]
    else:
        conf = conf[dataset_name]
    conf["dataset"] = dataset_name
    conf["model"] = paras["model"]

    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    device = torch.device("cuda:"+conf["gpu"] if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    dataset = Datasets(conf)    

    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items    

    conf['keepRate'] = paras['keepRate']
    conf['ris_adj_lambda'] = paras['ris_adj_lambda']
    conf['ris_lambda'] = paras['ris_lambda']
    conf['gnn_layer'] = paras['gnn_layer']


    print(conf)

    for lr, l2_reg, item_level_ratio, bundle_level_ratio, bundle_agg_ratio, embedding_size, num_layers, c_lambda, c_temp in \
            product(conf['lrs'], conf['l2_regs'], conf['item_level_ratios'], conf['bundle_level_ratios'], conf['bundle_agg_ratios'], conf["embedding_sizes"], conf["num_layerss"], conf["c_lambdas"], conf["c_temps"]):
        log_path = "./log/%s/%s" %(conf["dataset"], conf["model"])
        run_path = "./runs/%s/%s" %(conf["dataset"], conf["model"])
        checkpoint_model_path = "./checkpoints/%s/%s/model" %(conf["dataset"], conf["model"])
        checkpoint_conf_path = "./checkpoints/%s/%s/conf" %(conf["dataset"], conf["model"])
        if not os.path.isdir(run_path):
            os.makedirs(run_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        if not os.path.isdir(checkpoint_model_path):
            os.makedirs(checkpoint_model_path)
        if not os.path.isdir(checkpoint_conf_path):
            os.makedirs(checkpoint_conf_path)

        conf["l2_reg"] = l2_reg
        conf["embedding_size"] = embedding_size

        settings = []
        if conf["info"] != "":
            settings += [conf["info"]]

        settings += [conf["aug_type"]]
        if conf["aug_type"] == "ED":
            settings += [str(conf["ed_interval"])]
        if conf["aug_type"] == "OP":
            assert item_level_ratio == 0 and bundle_level_ratio == 0 and bundle_agg_ratio == 0

        settings += ["Neg_%d" %(conf["neg_num"]), str(conf["batch_size_train"]), str(lr), str(l2_reg), str(embedding_size)]

        conf["item_level_ratio"] = item_level_ratio
        conf["bundle_level_ratio"] = bundle_level_ratio
        conf["bundle_agg_ratio"] = bundle_agg_ratio
        conf["num_layers"] = num_layers
        settings += [str(item_level_ratio), str(bundle_level_ratio), str(bundle_agg_ratio), str(num_layers)]

        conf["c_lambda"] = c_lambda
        conf["c_temp"] = c_temp


        settings += [str(c_lambda), str(c_temp)]

        setting = "_".join(settings)
        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting
            
        run = SummaryWriter(run_path)

        # -- begin add define u-b diffusion model 
        u_bDiffusion_model = GaussianDiffusion(paras['noise_scale'], paras['noise_min'], paras['noise_max'], paras['steps'], conf).to(device)
        u_b_out_dims = eval(paras['dims']) + [conf['train_bundle']] #[1000,4771]
        u_b_in_dims = u_b_out_dims[::-1] #[4771,1000]
        u_bDenoise_model = Denoise(u_b_in_dims, u_b_out_dims, paras['d_emb_size'], conf, paras['norm']).to(device)
        u_bDenoise_opt = torch.optim.Adam(u_bDenoise_model.parameters(), lr=lr, weight_decay=0)

        # -- end add define u-b diffusion model 

        # -- begin add define u-i diffusion model
        u_iDiffusion_model = GaussianDiffusion(paras['noise_scale'], paras['noise_min'], paras['noise_max'], paras['steps'], conf).to(device)
        u_i_out_dims = eval(paras['dims']) + [conf['uiGraph_item']] #[1000,32770]
        u_i_in_dims = u_i_out_dims[::-1] #[32770,1000]
        u_iDenoise_model = Denoise(u_i_in_dims, u_i_out_dims, paras['d_emb_size'], conf, paras['norm']).to(device)
        u_iDenoise_opt = torch.optim.Adam(u_iDenoise_model.parameters(), lr=lr, weight_decay=0)
        # -- end add define u-i diffusion model


        # model
        if conf['model'] == 'DSCBR':
            model = DSCBR(conf, dataset.graphs).to(device)
        else:
            raise ValueError("Unimplemented model %s" %(conf["model"]))

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=conf["l2_reg"])

        batch_cnt = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])
        ed_interval_bs = int(batch_cnt * conf["ed_interval"])

        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0
        for epoch in range(conf['epochs']):
            epoch_anchor = epoch * batch_cnt
            model.train(True)
            epoch_start_time = time.time()
            print(f"Epoch {epoch+1} start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

            # --- begin add UB diffusion
            ubStart_time = time.time()
            ubDiffusionLoader = dataset.ub_diffusionLoader
            
            for i, batch in enumerate(ubDiffusionLoader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.to(device), batch_index.to(device)

                bEmbeds = model.bundles_feature.detach()
                uEmbeds = model.users_feature.detach()

                u_bDenoise_opt.zero_grad()

                diff_loss_ub, gc_loss_ub = u_bDiffusion_model.training_losses(u_bDenoise_model, batch_item, bEmbeds, batch_index)
                loss_ub = diff_loss_ub.mean() + gc_loss_ub.mean() * paras['e_loss']

                loss_ub.backward()

                u_bDenoise_opt.step()
                
                print('u_bDiffusion Step %d/%d' % (i, ubDiffusionLoader.dataset.__len__() // conf['batch_size_train']))
                print("Epoch: %d, Step %d/%d:, loss_ub: %.4f" %(epoch, i, len(ubDiffusionLoader), loss_ub.item()))

            #计算并打印batch运行时间
            ubEnd_time = time.time()
            print(f"UB-Diffusion total time: {ubEnd_time-ubStart_time:.2f} seconds")

            print('=========Start to re-build UB matrix========')
            
            with torch.no_grad():
                u_list_ubGraph = []
                b_list_ubGraph = []
                edge_list_ubGraph = []

                for _, batch in enumerate(ubDiffusionLoader):
                    ub_batch_item, ub_batch_index = batch
                    ub_batch_item, ub_batch_index = ub_batch_item.to(device), ub_batch_index.to(device)

                    # UB
                    ubDenoised_batch = u_bDiffusion_model.p_sample(u_bDenoise_model, ub_batch_item, paras['sampling_steps'], paras['sampling_noise'])
                    top_item, ub_indices_ = torch.topk(ubDenoised_batch, k=paras['rebuild_k'])

                    for i in range(ub_batch_index.shape[0]):
                        for j in range(ub_indices_[i].shape[0]):
                            u_list_ubGraph.append(int(ub_batch_index[i].cpu().numpy()))
                            b_list_ubGraph.append(int(ub_indices_[i][j].cpu().numpy()))
                            edge_list_ubGraph.append(1.0)
                    
                
                # UB 重新构建的图
                u_list_ubGraph = np.array(u_list_ubGraph)
                b_list_ubGraph = np.array(b_list_ubGraph)
                edge_list_ubGraph = np.array(edge_list_ubGraph)
                UB_matrix = dataset.buildUIMatrix(u_list_ubGraph, b_list_ubGraph, edge_list_ubGraph, type='UB')
                UB_matrix = u_bDiffusion_model.edgeDropper(UB_matrix)
            
            print('==========UB matrix built!=================')
            # --- end add UB diffusion

            # --- begin add UI diffusion
            # 记录ui batch开始时间
            uiStart_time = time.time()
            uiDiffusionLoader = dataset.ui_diffusionLoader
            for i, batch in enumerate(uiDiffusionLoader):
                

                ui_batch_item, ui_batch_index = batch
                ui_batch_item, ui_batch_index = ui_batch_item.to(device), ui_batch_index.to(device)
 
                iEmbeds = model.items_feature.detach() #(32770,128)
                uEmbeds = model.users_feature.detach()

                u_iDenoise_opt.zero_grad()

                diff_loss_ui, gc_loss_ui = u_iDiffusion_model.training_losses(u_iDenoise_model, ui_batch_item, iEmbeds, ui_batch_index)
                loss_ui = diff_loss_ui.mean() + gc_loss_ui.mean() * paras['e_loss']

                loss_ui.backward()

                u_iDenoise_opt.step()
                
                print('uiDiffusion Step %d/%d' % (i, uiDiffusionLoader.dataset.__len__() // conf['batch_size_train']))
                print("Epoch: %d, Step %d/%d:, loss_ui: %.4f" %(epoch, i, len(uiDiffusionLoader), loss_ui.item()))

            #计算并打印运行时间
            uiEnd_time = time.time()
            print(f"UI-Diffusion total time: {uiEnd_time-uiStart_time:.2f} seconds")

            print('============Start to re-build UI matrix=========')
            with torch.no_grad():
                u_list_uiGraph = []
                i_list_uiGraph = []
                edge_list_uiGraph = []

                for _, batch in enumerate(uiDiffusionLoader):
                    ui_batch_item, ui_batch_index = batch
                    ui_batch_item, ui_batch_index = ui_batch_item.to(device), ui_batch_index.to(device)

                    # UI
                    uiDenoised_batch = u_iDiffusion_model.p_sample(u_iDenoise_model, ui_batch_item, paras['sampling_steps'], paras['sampling_noise'])
                    top_item, ui_indices_ = torch.topk(uiDenoised_batch, k=paras['rebuild_k'])

                    for i in range(ui_batch_index.shape[0]):
                        for j in range(ui_indices_[i].shape[0]):
                            u_list_uiGraph.append(int(ui_batch_index[i].cpu().numpy()))
                            i_list_uiGraph.append(int(ui_indices_[i][j].cpu().numpy()))
                            edge_list_uiGraph.append(1.0)
                    
                
                # UI 重新构建的图
                u_list_uiGraph = np.array(u_list_uiGraph) #(8039,)
                i_list_uiGraph = np.array(i_list_uiGraph) #(8039,)
                edge_list_uiGraph = np.array(edge_list_uiGraph) #(8039,)
                UI_matrix = dataset.buildUIMatrix(u_list_uiGraph, i_list_uiGraph, edge_list_uiGraph, type='UI')
                UI_matrix = u_iDiffusion_model.edgeDropper(UI_matrix)
            
            print('=================UI matrix built!=============')
            # --- end add UI diffusion

            # 记录model 开始时间
            mainModelStart_time = time.time()
            for batch_i, batch in enumerate(dataset.train_loader):
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i

                ED_drop = False
                if conf["aug_type"] == "ED" and (batch_anchor+1) % ed_interval_bs == 0:
                    ED_drop = True

                # bpr_loss, c_loss, newCl_loss = model(batch, dataset.torchUIAdj, dataset.torchUBAdj_train, UI_matrix, UB_matrix, ED_drop=ED_drop)   # 执行model，返回loss 
                # loss = bpr_loss + conf["c_lambda"] * (c_loss + conf["scl_lambdas"] * newCl_loss)
                bpr_loss, c_loss = model(batch, dataset.torchUIAdj, dataset.torchUBAdj_train, UI_matrix, UB_matrix, ED_drop=ED_drop) 
                loss = bpr_loss + conf["c_lambda"] * (c_loss)
                loss.backward()
                optimizer.step()

                loss_scalar = loss.detach()
                bpr_loss_scalar = bpr_loss.detach()
                c_loss_scalar = c_loss.detach()
                run.add_scalar("loss_bpr", bpr_loss_scalar, batch_anchor)
                run.add_scalar("loss_c", c_loss_scalar, batch_anchor)
                run.add_scalar("loss", loss_scalar, batch_anchor)

                print("epoch: %d, loss: %.4f, bpr_loss: %.4f, c_loss: %.4f" %(epoch, loss_scalar, bpr_loss_scalar, c_loss_scalar))
                # print("epoch: %d, loss: %.4f, bpr_loss: %.4f, c_loss: %.4f, newCl_loss: %.4f" %(epoch, loss_scalar, bpr_loss_scalar, c_loss_scalar, newCl_loss.detach()))

                if (batch_anchor+1) % test_interval_bs == 0:  
                    metrics = {}
                    metrics["val"] = test(model, dataset.torchUIAdj, dataset.torchUBAdj_val, UI_matrix, UB_matrix, dataset.val_loader, conf)
                    metrics["test"] = test(model, dataset.torchUIAdj, dataset.torchUBAdj_test, UI_matrix, UB_matrix, dataset.test_loader, conf)                    
                    best_metrics, best_perform, best_epoch = log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch)
            #计算并打印运行时间
            mainModelEnd_time = time.time()
            print(f"mainModel total time: {mainModelEnd_time-mainModelStart_time:.2f} seconds")

            #计算并打印epoch运行时间
            epoch_end_time = time.time()
            print(f"Epoch {epoch+1} end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            print(f"Epoch {epoch+1} duration: {epoch_end_time - epoch_start_time:.2f} seconds")


def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform


def write_log(run, log_path, topk, step, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" %(m, topk), val_score[topk], step)
        run.add_scalar("%s_%d/Test" %(m, topk), test_score[topk], step)

    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" %(curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" %(curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk])

    log = open(log_path, "a")
    log.write("%s\n" %(val_str))
    log.write("%s\n" %(test_str))
    log.close()

    print(val_str)
    print(test_str)


def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch):
    for topk in conf["topk"]:
        write_log(run, log_path, topk, batch_anchor, metrics)

    log = open(log_path, "a")

    topk_ = 20
    print("top%d as the final evaluation standard" %(topk_))
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > best_metrics["val"]["ndcg"][topk_]:
        torch.save(model.state_dict(), checkpoint_model_path)
        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f" %(curr_time, best_epoch, topk, best_metrics["test"]["recall"][topk], best_metrics["test"]["ndcg"][topk])
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f" %(curr_time, best_epoch, topk, best_metrics["val"]["recall"][topk], best_metrics["val"]["ndcg"][topk])
            print(best_perform["val"][topk])
            print(best_perform["test"][topk])
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")

    log.close()

    return best_metrics, best_perform, best_epoch


def test(model, UIAdj, UBAdj, UI_matrix, UB_matrix, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.propagate(UIAdj, UBAdj, UI_matrix, UB_matrix, test=True)
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        pred_b = model.evaluate(rs, users.to(device))
        pred_b -= 1e8 * train_mask_u_b.to(device)
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b.to(device), pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float, device=device)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float, device=device)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()
