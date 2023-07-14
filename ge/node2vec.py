# -*- coding:utf-8 -*-

"""



Author:

    Weichen Shen,wcshen1994@163.com



Reference:

    [1] Grover A, Leskovec J. node2vec: Scalable feature learning for networks[C]//Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2016: 855-864.(https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf)



"""

from gensim.models import Word2Vec
import pandas as pd

from ge.walker import RandomWalker
import numpy as np


class Node2Vec:

    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1):

        self.graph = graph
        self._embeddings = {}
        self.walker = RandomWalker(graph, p=p, q=q, )

        # print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()

        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)
        # self.network_analysis()

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        # print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        # print("Learning embedding vectors done!")

        self.w2v_model = model

        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv.get_vector(word)

        return self._embeddings

    # def network_analysis(self):
    #     '''
    #         Input: directed graph
    #         Output: Number of Nodes, Multi-relational Adjacency Matrix 
    #     '''
    #     print(self.graph.nodes)
    #     print(self.graph.edges)
    #     num_nodes = len(self.graph.nodes)
    #     adj_matrix = np.zeros([3, num_nodes, num_nodes], np.float64)
    #     counter = 0
    #     for edge in self.graph.edges:
    #         counter += 1
    #         if counter > 5:
    #             break
    #         print(type(self.graph[edge[0]][edge[1]]), self.graph[edge[0]][edge[1]])
    #     # print(adj_matrix.shape, adj_matrix)
    #     return num_nodes
