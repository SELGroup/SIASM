import glob
import networkx as nx
import numpy as np
from tqdm import tqdm

def sim_graph(n_legit_users=50, RANDOM_FOLLOW=3):
    N = n_legit_users + 1
    mat = np.zeros((N, N))
    idx = np.random.choice(n_legit_users, (3+2+1)*RANDOM_FOLLOW, replace=False)
    idx1 = idx[0:3]
    idx2 = idx[3:5]
    idx3 = idx[5:6]
    for i in idx:
        mat[i, idx2] = 1
        for j in idx2:
            mat[j, idx3] = 1
    g = nx.convert_matrix.from_numpy_matrix(mat, create_using=nx.DiGraph)
    return mat, g


def get_embeds(g, node_embed_dim=2, win=8, emb_iters=8, num_walks=8, walk_len=16, cpu=2, alg="deepwalk", p=1, q=1):
    d={}
    for n in g.nodes:
        d[n]=str(n)
    g1 = nx.relabel_nodes(g,d)
    if alg == "deepwalk":
        from ge.deepwalk import DeepWalk
        graph_model = DeepWalk(g1, num_walks=num_walks, walk_length=walk_len, workers=cpu)
        graph_model.train(window_size=win, iter=emb_iters, embed_size=node_embed_dim)

    elif alg == "node2vec":
        from ge.node2vec import Node2Vec
        graph_model = Node2Vec(g1, num_walks=num_walks, walk_length=walk_len, workers=cpu, p=p, q=q)
        graph_model.train(embed_size=node_embed_dim, window_size=win, workers=cpu, iter=emb_iters)
    # elif alg == "dgi":
    #     from ge.dgi import DGI_Model
    #     # node_embed_dim
    #     graph_model = DGI_Model(g1, hid_units=node_embed_dim)
    #     graph_model.train()

    emb1 = graph_model.get_embeddings()
    # if alg == 'dgi':
    #     embs = []
    #     for i in range(emb1.shape[0]):
    #         embs.append(emb1[i].cpu().numpy())
    embs = {}
    for n in emb1.keys():
        embs[int(n)] = emb1[n]
    embs = [embs[k] for k in range(len(g.nodes))]
    return embs

    # print(g.edges())
    # embs = get_embeds(g)
    # print(embs)

    # graphs = []
    # files = glob.glob('../../../notebook/*.gpickle')
    # for file in files:
    #     g = nx.read_gpickle(file)
    #     graphs.append(g)

    # g = graphs[0]
    # embs = get_embeds(g)
