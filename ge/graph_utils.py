import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
# from webweb import Web
import networkx as nx
import numpy as np
from community import community_louvain as community
import pickle
from itertools import product
import time
import os
import getpass
import random


def randomize_graph(graph_name='advbot_train', k=1000, mode=1):
    out_file = './database/{}_community_stats.pkl'.format(graph_name)
    if not os.path.exists(out_file):
        path = './database/{}.pkl'.format(graph_name)
        g = open_g(path)
        comm, szs, p_in, p_btw = get_comm_stats(g.to_undirected(), verbose=False)
        total = len(g)
        pickle.dump([total, szs, p_in, p_btw], open(out_file, 'wb'))
    else:
        total, szs, p_in, p_btw = pickle.load(open(out_file, 'rb'))

    if mode == 1:
        graph = gen_sim_graph(szs, p_in, p_btw, k=k)
    
    elif mode == 2:
        graph = gen_rt_graph(szs, p_in, p_btw, k=k)

    elif mode == 3:
        graph = random_star(szs, p_in, p_btw, case=5)
    return graph


def random_star(szs, p_in, p_btw, k=1, case=1):
    np.random.seed(int(time.time()))

    if case == 1:
        a = np.random.randint(low=15, high=25)
        b = np.random.randint(low=10, high=15)
        c = np.random.randint(low=5, high=10)
        s = [a,b,c]
        s += [2]*(50-np.sum(s))
        g = gen_rt_graph(s, p_in, p_btw, k=100)
        for i in range(70-len(g)):
            g.add_node(len(g))
        

    elif case == 2:
        a = np.random.randint(low=30, high=50)
        b = np.random.randint(low=20, high=40)
        c = np.random.randint(low=10, high=25)
        d = np.random.randint(low=5, high=10)
        s = [a,b,c,d]
        s += [3]*(110-np.sum(s))
        g = gen_rt_graph(s, p_in, p_btw, k=100)
        for i in range(200-len(g)):
            g.add_node(len(g))

    elif case == 3:
        s = []
        max_node = 500
        random.shuffle(szs)
        num_comm = np.random.randint(low=1, high=5)
        for q in szs[:num_comm]:
            a = np.random.randint(low=q-10, high=q+10)
            s += [a]
        s += [1]*(300-np.sum(s))
        g = gen_rt_graph(s, p_in, p_btw, k=100)
        for i in range(max_node-len(g)):
            g.add_node(len(g))

    elif case == 5:
        max_node = 1500
        s = []
        random.shuffle(szs)
        num_comm = np.random.randint(low=1, high=5)
        for q in szs[:num_comm]:
            a = np.random.randint(low=q-10, high=q+10)
            s += [a]
        s += [3]*(400-np.sum(s))
        g = gen_rt_graph(s, p_in, p_btw, k=100)
        for i in range(max_node-len(g)):
            g.add_edge(np.random.choice(len(g)), len(g))

    elif case == 4:
        N = 500
        g = nx.DiGraph()
        for i in range(N):
            g.add_node(i)
        idx = np.random.choice(N, (10+2+1), replace=False)
        idx1 = idx[0:10]
        idx2 = idx[10:12]
        idx3 = idx[12:13]
        for j in idx2:
            for i in idx1:
                g.add_edge(j, i)
        for j in idx3:
            for i in idx2:
                g.add_edge(j, i)

    g = nx.convert_node_labels_to_integers(g)
    nodes = list(g.nodes)
    rnd_idx = np.random.permutation(nodes)
    mapping = {nodes[i]:rnd_idx[i] for i in range(len(nodes))}
    g = nx.relabel_nodes(g, mapping)

    return g


def load_test_graph(max_node, community_based=False, graph_name="copen"):
    original_g = open_g('/home/{}/Documents/graph_sample_rl/data/rt/{}.pkl'.format(getpass.getuser(), graph_name))
    comm, szs, p_in, p_btw = get_comm_stats(original_g, verbose=False)
    if not community_based:
        new_g = cut_graph(original_g, max_node=max_node)
    else:
        new_g = cut_graph2(original_g, comm, max_node=max_node)
    return new_g


def cut_graph2(g, comm, max_node = 288):
    direct_g = nx.DiGraph()
    szs = [len(comm[c]) for c in list(comm.keys())]
    szs = np.argsort(szs)[::-1]
    stop = False
    for i in szs:
        idx = list(comm.keys())[i]
        nodes = comm[idx]
        for edge in g.edges(nodes):
            if g.degree(edge[0]) < g.degree(edge[1]):
                direct_g.add_edge(edge[1], edge[0])
            else:
                direct_g.add_edge(edge[0], edge[1])
            if len(direct_g) > max_node:
                stop = True
                break
        if stop:
            break
    direct_g = nx.convert_node_labels_to_integers(direct_g)
    return direct_g


def cut_graph(original_g, max_node = 288):
    direct_g = nx.DiGraph()
    for edge in original_g.edges():
        if original_g.degree(edge[0]) < original_g.degree(edge[1]):
            direct_g.add_edge(edge[1], edge[0])
        else:
            direct_g.add_edge(edge[0], edge[1])

    new_g = nx.DiGraph()
    out_degree = [direct_g.out_degree(i) for i in range(len(direct_g))]
    idx = np.argsort(out_degree)[::-1]
    stop = False
    for i in idx:
        edges = direct_g.in_edges(i)
        for edge in edges:
            new_g.add_edge(edge[0], edge[1])
            if len(new_g) > max_node:
                stop = True
        if stop:
            break
    new_g = nx.convert_node_labels_to_integers(new_g)
    return new_g


def open_g(path):
    with open(path,'rb') as fl:
        g = pickle.load(fl)
    return g


def get_comm_stats(g, verbose=True):
    n, m = g.number_of_nodes(), g.number_of_edges()
    if verbose:
        print("Nodes: %d, Edges: %d, Average Degree: %f"%(n,m,2*m/n))
    partition = community.best_partition(g)
    communities = set(partition.values())
    nodes = {c: [n for n in partition.keys() if partition[n]==c] for c in communities}
    szs = [len(nodes[c]) for c in nodes.keys()]
    if verbose:
        print(sorted(szs))
        print("Number of communities: %d, Stdev: %f"%(len(communities),np.std(szs)))
    f1 = lambda x : (x*(x-1))/2
    f2 = lambda x,y : x*y

    max_in_edges = np.sum([f1(len(nodes[n])) for n in nodes.keys()])
    max_btw_edges = 0
    for i in communities:
        j = i+1
        while j<len(communities):
            max_btw_edges+= len(nodes[i]) * len(nodes[j])
            j+=1
    
    in_edges, btw_edges = 0,0
    for u,v in g.edges():
        if partition[u] == partition[v]:
            in_edges+= 1
        else:
            btw_edges+=1
    p_in, p_btw = in_edges/max_in_edges,btw_edges/max_btw_edges
    if verbose:
        print("p_in:%f, p_btw:%f"%(p_in, p_btw))
    return nodes, sorted(szs), p_in, p_btw


def gen_sim_graph(szs, p_in, p_btw, k=1000):
    gen_gr = nx.random_partition_graph(szs[::-1][:k], p_in, p_btw, directed=True)
    return gen_gr



def gen_rt_graph(szs, p_in, p_btw, iters=1, k=1):
    np.random.seed(int(time.time()))
    szs = np.sort(szs)[::-1][:k]
    gen_gr = nx.DiGraph()
    nt = 0
    par = {}
    for i,s in enumerate(szs):
        par[i] = list()
        for _ in range(s):
            par[i].append(nt)
            nt +=1
    for i,s in enumerate(szs):
        u = par[i][0]
        if len(par[i])>0:
            for v in par[i][1:]: gen_gr.add_edge(u, v)

    for s1,s2 in product(par.keys(),par.keys()):
        if s1>=s2: continue
        for u,v in product(par[s1],par[s2]):
            if np.random.rand()<p_btw:
                gen_gr.add_edge(u, v)
    gen_gr = nx.convert_node_labels_to_integers(gen_gr)
    return gen_gr






