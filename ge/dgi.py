import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from ge.models import DGI, LogReg
from ge.util import process
import networkx as nx
from scipy.sparse import lil_matrix

class DGI_Model:

    def __init__(self, graph, batch_size=1, nb_epoches=10000, patience=20, lr=0.001,
     l2_coef=0.0, drop_prob=0.0, hid_units=512, sparse=True, nonlinearity = 'prelu'):
        self.graph = graph
        self.batch_size = batch_size
        self.nb_epoches = nb_epoches
        self.patience = patience
        self.lr = lr
        self.l2_coef = l2_coef
        self.drop_prob = drop_prob
        self.hid_units = hid_units
        self.sparse = sparse
        self.nonlinearity = nonlinearity
        features = lil_matrix(np.eye(len(self.graph.nodes)))
        self.features, _ = process.preprocess_features(features)
        self.nb_nodes = self.features.shape[0]
        self.ft_size = self.features.shape[1]
        adj = nx.to_scipy_sparse_array(self.graph).tocsr()
        self.adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        self.model = DGI(self.ft_size, self.hid_units, self.nonlinearity)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_coef)
    
    def train(self):
        if self.sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(self.adj) 
        else:
            adj = (self.adj + sp.eye(self.adj.shape[0])).todense()
        features = torch.FloatTensor(self.features[np.newaxis])
        if not self.sparse:
            adj = torch.FloatTensor(self.adj[np.newaxis])
        if torch.cuda.is_available():
            print('Using CUDA to train DGI')
            self.model.cuda()
            features=features.cuda()
            if self.sparse:
                sp_adj = sp_adj.cuda()
            else:
                adj = adj.cuda()
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0
        best_weights = None
        for epoch in range(self.nb_epoches):
            self.model.train()
            self.optimiser.zero_grad()

            idx = np.random.permutation(self.nb_nodes)
            shuf_fts = features[:, idx, :]

            lbl_1 = torch.ones(self.batch_size, self.nb_nodes)
            lbl_2 = torch.zeros(self.batch_size, self.nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)

            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda()
                lbl = lbl.cuda()
            
            logits = self.model(features, shuf_fts, sp_adj if self.sparse else adj, self.sparse, None, None, None) 

            loss = b_xent(logits, lbl)

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                # torch.save(self.model.state_dict(), 'best_dgi.pkl')
                best_weights = self.model.state_dict()
            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                print('Early stopping!')
                break

            loss.backward()
            self.optimiser.step()

        print('Loading the DGI model of {}th epoch'.format(best_t))
        # self.model.load_state_dict(torch.load('best_dgi.pkl'))
        self.model.load_state_dict(best_weights)
        self.features = features
        if self.sparse:
            self.sp_adj = sp_adj
        else:
            self.adj = adj

    def get_embeddings(self):
        embeds, _ = self.model.embed(self.features, self.sp_adj if self.sparse else self.adj, self.sparse, None)
        return embeds[0]
