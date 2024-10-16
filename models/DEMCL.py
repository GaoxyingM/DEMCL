#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 

from loss1 import DCL
from loss1.dcl import DCLW


def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)



    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]

    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class DEMCL(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.num_positive=conf["positive_t"]

        self.init_emb()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        """
        #(好像没用) generate the graph without any dropouts for testing
        self.get_item_level_graph_ori()
        self.get_bundle_level_graph_ori()
        self.get_bundle_agg_graph_ori()

        #(好像没用) generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.get_item_level_graph()
        self.get_bundle_level_graph()
        self.get_bundle_agg_graph()

        ## cal    b-b graph  用以获得bundle间的相似度
        self.bb_graph = self.get_ii_constraint_mat(self.ub_graph, 3)
        """
        self.get_bundle_agg_graph_ori()
        self.get_bundle_agg_graph()
        self.init_md_dropouts()

        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]
        self.kk=10

        # --- begin add gcn (used to calculate cl_MM)
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.conf['gnn_layer'])])
        # --- end add gcn (used to calculate cl_MM)


    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.bundle_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.bundle_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def get_item_level_graph(self):
        ui_graph = self.ui_graph
        device = self.device
        modification_ratio = self.conf["item_level_ratio"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = item_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.item_level_graph = to_tensor(laplace_transform(item_level_graph)).to(device)


    def get_item_level_graph_ori(self):
        ui_graph = self.ui_graph
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        self.item_level_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)


    def get_bundle_level_graph(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.bundle_level_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)


    def get_bundle_level_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.bundle_level_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)


    def get_bundle_agg_graph(self):
        bi_graph = self.bi_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)


    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)


    def get_ii_constraint_mat(self,train_mat, num_neighbors, ii_diagonal_zero=True):

        print('Computing \\Omega for the item-item graph... ')
        A = train_mat.T.dot(train_mat)  # B * B
        n_items = A.shape[0]
        res_mat = torch.zeros((n_items, num_neighbors))
        res_sim_mat = torch.zeros((n_items, num_neighbors))
        if ii_diagonal_zero:
            A[range(n_items), range(n_items)] = 0

        for i in range(n_items):
            row =   torch.from_numpy(A.getrow(i).toarray()[0])
            row_sims, row_idxs = torch.topk(row, num_neighbors)
            res_mat[i] = row_idxs
            res_sim_mat[i] = row_sims
            if i % 15000 == 0:
                print('i-i constraint matrix {} ok'.format(i))

        print('Computation \\Omega OK!')
        return res_mat.long()


    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test: # !!! important
                features = mess_dropout(features)

            features = features / (i+2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature


    def get_IL_bundle_rep(self, IL_items_feature, test):
        if test:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph_ori, IL_items_feature)
        else:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph, IL_items_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_bundles_feature = self.bundle_agg_dropout(IL_bundles_feature)

        return IL_bundles_feature


    def propagate(self, adjUI, adjUB, geneUI_matrix, geneUB_matrix, test=False):
        # ========= 1. Generated Item Level ==============
        embedsUIAdj = torch.cat([self.users_feature, self.items_feature])
        embedsUIAdj = F.normalize(torch.spmm(geneUI_matrix, embedsUIAdj), p=2, dim=1)

        embedsUI = torch.cat([self.users_feature, self.items_feature])
        embedsUI = F.normalize(torch.spmm(adjUI, embedsUI), p=2, dim=1)

        embedsUI_ = torch.cat([embedsUI[:self.conf["num_users"]], self.items_feature])
        embedsUI_ = F.normalize(torch.spmm(adjUI, embedsUI_), p=2, dim=1)
        embedsUI += embedsUI_

        embedsUI += self.conf['ris_adj_lambda'] * embedsUIAdj

        ## obtain User embedding and Bundel embedding 
        IL_users_feature = embedsUI[:self.conf['num_users']]
        IL_bundles_feature = self.get_IL_bundle_rep(embedsUI[self.conf['num_users']:], test)

        # ========= 2. Generated Bundle Level ==============
        embedsUBAdj = torch.cat([self.users_feature, self.bundles_feature])
        embedsUBAdj = F.normalize(torch.spmm(geneUB_matrix, embedsUBAdj), p=2, dim=1)

        embedsUB = torch.cat([self.users_feature, self.bundles_feature])
        embedsUB = F.normalize(torch.spmm(adjUB, embedsUB), p=2, dim=1)

        embedsUB_ = torch.cat([embedsUB[:self.conf["num_users"]], self.bundles_feature])
        embedsUB_ = F.normalize(torch.spmm(adjUB, embedsUB_), p=2, dim=1)
        embedsUB += embedsUB_

        embedsUB += self.conf['ris_adj_lambda'] * embedsUBAdj

        ## obtain User embedding and Bundel embedding 
        BL_users_feature = embedsUB[:self.conf['num_users']]
        BL_bundles_feature = embedsUB[self.conf['num_users']:]

        ## ============ 3. 汇总 =====   
        users_feature = [IL_users_feature, BL_users_feature]
        bundles_feature = [IL_bundles_feature, BL_bundles_feature]

        return users_feature, bundles_feature

        """
    
        #  =============================  item level propagation  =============================
        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph_ori, self.users_feature, self.items_feature, self.item_level_dropout, test)
        else:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph, self.users_feature, self.items_feature, self.item_level_dropout, test)

        # aggregate the items embeddings within one bundle to obtain the bundle representation
        IL_bundles_feature = self.get_IL_bundle_rep(IL_items_feature, test)

        #  ============================= bundle level propagation =============================
        if test:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph_ori, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)
        else:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)

        users_feature = [IL_users_feature, BL_users_feature]
        bundles_feature = [IL_bundles_feature, BL_bundles_feature]

        return users_feature, bundles_feature
        """

    def FN_mask(self,sim):
        # mask(Removing False Negative Samples)
        max_ = torch.max(sim, dim=-1, keepdim=True).values
        min_ = torch.min(sim, dim=-1, keepdim=True).values
        sim_ = (sim - min_) / (max_ - min_)
        eye_matrix=torch.eye(sim.shape[0],dtype=torch.long).to("cuda:0")
        sim_[eye_matrix==1]=0

        return sim_>=1.0

    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)

        pos_score = torch.sum(pos * aug, dim=1)  # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0))  # [batch_size, batch_size]

 # 过滤 false  negative
        # m_1 = self.FN_mask(ttl_score)
        # ttl_score[m_1 == 1] = float("-inf")
#-----------------------------------------------

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]
        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss
    def scl(self,pred):
        # pred: [bs, 1+neg_num]
        if pred.shape[1] > 2:
            negs = pred[:, 1:]
            pos = pred[:, 0]
        else:
            negs = pred[:, 1]
            pos = pred[:, 0]
        

        negs=torch.exp(negs/self.c_temp).sum(dim=-1)

        pos=torch.exp(pos/self.c_temp)
        loss = (- torch.log(pos / (pos + negs))).mean()

        return loss

    def cal_loss(self, users_feature, bundles_feature):
        # IL: item_level, BL: bundle_level
        # [bs, 1, emb_size]
        IL_users_feature, BL_users_feature = users_feature
        # [bs, 1+neg_num, emb_size]
        IL_bundles_feature, BL_bundles_feature = bundles_feature

        # cal  cross view cl_loss
        u_cross_view_cl = self.cal_c_loss(IL_users_feature, BL_users_feature)
        b_cross_view_cl = self.cal_c_loss(IL_bundles_feature, BL_bundles_feature)

        c_losses = [u_cross_view_cl, b_cross_view_cl]

        c_loss = sum(c_losses) / len(c_losses)

        return c_loss

    # --- begin (update)only use generated matrix 
    def forward_cl_MM(self, adjUI, adjUB, geneUI_matirx, geneUB_matrix):
        # UI graph
        embedsUI = torch.cat([self.users_feature, self.items_feature])
        embedsUI = F.normalize(torch.spmm(geneUI_matirx, embedsUI), p=2, dim=1)

        embeds1 = embedsUI
        embedsLst1 = [embeds1]
        for gcn in self.gcnLayers:
            embeds1 = F.normalize(gcn(geneUI_matirx, embedsLst1[-1]), p=2, d=1)
            embedsLst1.append(embeds1)
        embeds1 = sum(embedsLst1)

        # aggregate the items embeddings within one bundle to obtain the bundle representation
        IL_bundles_feature = self.get_IL_bundle_rep(embeds1[self.conf['num_users']:], test=False)
        
        # UB graph
        embedsUB = torch.cat([self.users_feature, self.bundles_feature])
        embedsUB = F.normalize(torch.spmm(geneUB_matrix, embedsUB), p=2, dim=1)

        embeds2 = embedsUB
        embedsLst2 = [embeds2]
        for gcn in self.gcnLayers:
            embeds2 = F.normalize(gcn(geneUB_matrix, embedsLst2[-1]), p=2, dim=1)
            embedsLst2.append(embeds2)
        embeds2 = sum(embedsLst2)

        return embeds1[:self.conf['num_users']], IL_bundles_feature, embeds2[:self.conf['num_users']], embeds2[self.conf['num_users']:]
    # --- end (update)only use generated matrix 

    """
    # new add
    # -- begin add contrastive learning
    def forward_cl_MM(self, adjUI, adjUB, geneUI_matirx, geneUB_matrix):
        # UI graph
        embedsUI = torch.concat([self.users_feature, self.items_feature])
        embedsUI = torch.spmm(geneUI_matirx, embedsUI)

        embeds1 = embedsUI
        embedsLst1 = [embeds1]
        for gcn in self.gcnLayers:
            embeds1 = gcn(adjUI, embedsLst1[-1])
            embedsLst1.append(embeds1)
        embeds1 = sum(embedsLst1)

        # aggregate the items embeddings within one bundle to obtain the bundle representation
        IL_bundles_feature = self.get_IL_bundle_rep(embeds1[self.conf['num_users']:], test=False)
        
        # UB graph
        embedsUB = torch.concat([self.users_feature, self.bundles_feature])
        embedsUB = torch.spmm(geneUB_matrix, embedsUB)

        embeds2 = embedsUB
        embedsLst2 = [embeds2]
        for gcn in self.gcnLayers:
            embeds2 = gcn(adjUB, embedsLst2[-1])
            embedsLst2.append(embeds2)
        embeds2 = sum(embedsLst2)

        return embeds1[:self.conf['num_users']], IL_bundles_feature, embeds2[:self.conf['num_users']], embeds2[self.conf['num_users']:]

    # -- end add contrastive learning
    """

    def forward(self, batch,adjUI, adjUB, diffusionUI_matrix, diffusionUB_matrix, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        """
        if ED_drop:
            self.get_item_level_graph()
            self.get_bundle_level_graph()
            self.get_bundle_agg_graph()
        """
        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, bundles = batch
        users_feature, bundles_feature = self.propagate(adjUI, adjUB, diffusionUI_matrix, diffusionUB_matrix)

        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]
        bundles_embedding = [i[bundles] for i in bundles_feature]

        c_loss = self.cal_loss(users_embedding, bundles_embedding)
        
        """
        # --- begin calculate new clLoss
        usrEmbedsUI, bunEmbedsUI, usrEmbedsUB, bunEmbedsUB = self.forward_cl_MM(adjUI, adjUB, diffusionUI_matrix, diffusionUB_matrix)
        ## 1. obtain user embedding and bundle embedding
        ### 1.1 Generated Item Level: obtain user embedding and bundle embedding in batch
        usrEmbedUI_batch = usrEmbedsUI[users].expand(-1,bundles.shape[1],-1) #(2048,101,128)
        bunEmbedUI_batch = bunEmbedsUI[bundles] ##(2048,101,128)

        ### 1.2 Generated Bundle Level: obtain user embedding and bundle embedding in batch
        usrEmbedUB_batch = usrEmbedsUB[users].expand(-1,bundles.shape[1],-1) #(2048,101,128)
        bunEmbedUB_batch = bunEmbedsUB[bundles] ##(2048,101,128)

       
        ## 2. cross-view contrastive, use multiply 
        ### 2.1. ori bundel view & generated item view
        pred_scl11 = torch.sum(users_embedding[1] * bunEmbedUI_batch, 2) #torch.Size([2048,101])
        pred_scl12 = torch.sum(usrEmbedUI_batch * bundles_embedding[1], 2)  #torch.Size([2048,101])
        scl_loss_ori1 = self.scl(pred_scl11)
        scl_loss_ori1 += self.scl(pred_scl12)

        ### 2.2. ori item view & generated bundel view
        pred_scl21 = torch.sum(usrEmbedUB_batch * bundles_embedding[0], 2)
        pred_scl22 = torch.sum(users_embedding[0] * bunEmbedUB_batch, 2)
        scl_loss_ori2 = self.scl(pred_scl21)
        scl_loss_ori2 += self.scl(pred_scl22)

        newsclLoss = scl_loss_ori1 + scl_loss_ori2
        # --- end calculate new clLoss
        """

        
        IL_users_feature, BL_users_feature = users_embedding
        # [bs, 1+neg_num, emb_size]
        IL_bundles_feature, BL_bundles_feature = bundles_embedding
        
        pred = torch.sum(IL_users_feature * IL_bundles_feature, 2) + torch.sum(BL_users_feature * BL_bundles_feature, 2)

        bpr_loss = cal_bpr_loss(pred)

        return bpr_loss, c_loss#, newsclLoss
        # return bpr_loss, c_loss, newsclLoss

    # Dynamic Negative Sampling First, a batch of negative samples are evenly sampled,
    # and then a few of the negative samples are selected to have the lowest similarity scores, or of course, the highest. (The higher the score, the harder it may be)

    def sample_neg(self,bundles,users_embedding,bundles_embedding,sample_num):
        users_embedding=users_embedding[:,1:,:]
        neg_bundles_embedding=bundles_embedding[:,1:,:]

        users_embedding = F.normalize(users_embedding, p=2, dim=2)
        neg_bundles_embedding = F.normalize(neg_bundles_embedding, p=2, dim=2)


        # item_candi=bundles[:,1:]

        sim_score=torch.sum(users_embedding * neg_bundles_embedding,2)
        sim_score = -sim_score
        topk= torch.topk(sim_score,3)



    '''
        Sampling according to probability
    '''
        # item_candi=item_candi.tolist()
        #
        # prob=torch.exp(torch.sum(users_embedding * neg_bundles_embedding, 2)) #(batch_size,neg_num)
        # prob_sum=torch.sum(prob,dim=1).unsqueeze(-1)
        # prob=(prob/prob_sum)**self.ia
        # prob_list=prob.tolist()
        #
        # neg_all = []
        #
        # for i in range(bundles.shape[0]):
        #     random_index_list = np.random.choice(item_candi[i], sample_num, prob_list[i])
        #     neg_all.append(random_index_list)


        # neg_all = torch.tensor(neg_all).to(self.device)
        # item_all=torch.cat([bundles[:,0].unsqueeze(-1),neg_all],1)
        # return item_all

    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result
        users_feature_atom, users_feature_non_atom = [i[users] for i in users_feature]
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature

        scores = torch.mm(users_feature_atom, bundles_feature_atom.t()) + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t())
        return scores



# --- begin add GCN
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()
    
    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)

# --- end add GCN