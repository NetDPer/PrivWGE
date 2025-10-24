import numpy as np
import tensorflow as tf
import time
from itertools import islice

from minibatch import build_batch_from_edges as build_batch
from graphsage import GraphSageUnsupervised as GraphSage
from config import ENABLE_UNKNOWN_OP
import networkx as nx
from collections import defaultdict
from sklearn import metrics
from sklearn.externals import joblib
from autodp import rdp_acct, rdp_bank
import argparse
import functions

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=512)
parser.add_argument('--RDP', default=True)
parser.add_argument('--delta', default=10**(-5))
parser.add_argument('--epsilon', default=6)
parser.add_argument('--eps_tolerance', default=0.01)
parser.add_argument('--indep_run_times', default=1)
parser.add_argument('--SMALL_BATCH_SIZE', default=5000)
parser.add_argument('--sigma', default=5)

args = parser.parse_args()  # parameter

#### NN parameters
SAMPLE_SIZES = [25, 10]
INTERNAL_DIM = 128
NEG_WEIGHT = 1.0

#### training parameters
BATCH_SIZE = 512
NEG_SIZE = 20
TRAINING_STEPS = 50
LEARNING_RATE = 0.01

def generate_training_minibatch(trainGraph, adj_mat_dict, batch_size, sample_sizes, neg_size):
    edges = np.array([(k, v) for k in adj_mat_dict for v in adj_mat_dict[k]])
    nodes = np.array(list(adj_mat_dict.keys()))
    while True:
        mini_batch_edges = edges[np.random.randint(edges.shape[0], size = batch_size), :]
        batch = build_batch(trainGraph, mini_batch_edges, nodes, adj_mat_dict, sample_sizes, neg_size)
        yield batch

def get_d_columns_adjacency_matrix(G, d):
    adj_sparse = nx.adjacency_matrix(G)

    if d is not None:
        # If only the first d columns are needed
        if d > adj_sparse.shape[1]:
            raise ValueError("d cannot be larger than the number of nodes in the graph.")
        adj_sparse = adj_sparse[:, :d]

    # Or convert to a numpy matrix
    dense_matrix = adj_sparse.todense()

    return dense_matrix

def load_dataset(file_name, train_filename, test_task, directed=False):
    if test_task == 'lwp':
        adj_lists = defaultdict(set)

        with open(file_name, 'r') as f:
            if directed:
                G = nx.DiGraph()
            else:
                G = nx.Graph()
            for line in f:
                edge = line.strip().split()
                if len(edge) == 3:
                    w = float(edge[2])
                else:
                    w = 1.0
                G.add_edge(int(edge[0]), int(edge[1]), weight=w)

        original_graph = G

        trainGraph = nx.adjacency_matrix(G)

        train_pos = joblib.load(train_filename + 'train_pos.pkl')
        train_neg = joblib.load(train_filename + 'train_neg.pkl')
        test_pos = joblib.load(train_filename + 'test_pos.pkl')
        test_neg = joblib.load(train_filename + 'test_neg.pkl')

        trainGraph = trainGraph.copy()  # the observed network
        trainGraph[test_pos[0], test_pos[1]] = 0  # mask test links
        trainGraph[test_pos[1], test_pos[0]] = 0  # mask test links
        trainGraph.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

        row, col = train_neg
        trainGraph = trainGraph.copy()
        trainGraph[row, col] = 1  # inject negative train
        trainGraph[col, row] = 1  # inject negative train

        # Convert the weighted adjacency matrix to a NetworkX graph
        G = nx.from_scipy_sparse_matrix(trainGraph)

        num_nodes = G.number_of_nodes()
        feat_data = get_d_columns_adjacency_matrix(G, 128)

        # Create a dictionary to store the adjacency list for each node (using NumPy arrays)
        adj_lists = {node: np.array(list(G.neighbors(node))) for node in G.nodes()}

        return num_nodes, feat_data, adj_lists, test_pos, test_neg, original_graph, G

    if test_task == 'StrucEqu':
        adj_lists = defaultdict(set)

        with open(file_name, 'r') as f:
            if directed:
                G = nx.DiGraph()
            else:
                G = nx.Graph()
            for line in f:
                edge = line.strip().split()
                if len(edge) == 3:
                    w = float(edge[2])
                else:
                    w = 1.0
                G.add_edge(int(edge[0]), int(edge[1]), weight=w)

        original_graph = G
        trainGraph = nx.adjacency_matrix(G)

        # Convert the weighted adjacency matrix to a NetworkX graph
        G = nx.from_scipy_sparse_matrix(trainGraph)
        num_nodes = G.number_of_nodes()
        feat_data = get_d_columns_adjacency_matrix(G, 128)

        # Create a dictionary to store the adjacency list for each node (using NumPy arrays)
        adj_lists = {node: np.array(list(G.neighbors(node))) for node in G.nodes()}

        return num_nodes, feat_data, adj_lists, None, None, original_graph, G

def CalcAUC_Unweighted(original_graph, sim, test_pos, test_neg):
    u_list = test_pos[0]
    v_list = test_pos[1]

    pos_label_list = []
    for u, v in zip(u_list, v_list):
        u_v_weight = original_graph.get_edge_data(u, v)['weight']
        pos_label_list.append(u_v_weight)

    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    # labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    labels = np.hstack([pos_label_list, np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return auc

def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return auc

def CalcMSE(graph, sim, test_pos, test_neg):
    u_list = test_pos[0]
    v_list = test_pos[1]

    pos_label_list = []
    for u, v in zip(u_list, v_list):
        u_v_weight = graph.get_edge_data(u, v)['weight']

        pos_label_list.append(u_v_weight)

    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    data_min = np.min(scores)
    data_max = np.max(scores)
    scores = (scores - data_min) / (data_max - data_min)

    labels = np.hstack([pos_label_list, np.zeros(len(neg_scores))])
    data_min = np.min(labels)
    data_max = np.max(labels)
    labels = (labels - data_min) / (data_max - data_min)

    mse_value = metrics.mean_squared_error(labels, scores)

    return mse_value

def CalcMSE2(graph, W_emb, test_pos, test_neg):
    u_list = test_pos[0]
    v_list = test_pos[1]

    pos_label_list = []
    for u, v in zip(u_list, v_list):
        u_v_weight = graph.get_edge_data(u, v)['weight']
        pos_label_list.append(u_v_weight)

    # Calculate the similarity of positive samples
    pos_scores = np.array([np.dot(W_emb[u], W_emb[v]) for u, v in zip(u_list, v_list)])

    # Calculate the similarity of negative samples
    neg_u_list = test_neg[0]
    neg_v_list = test_neg[1]
    neg_scores = np.array([np.dot(W_emb[u], W_emb[v]) for u, v in zip(neg_u_list, neg_v_list)])

    scores = np.concatenate([pos_scores, neg_scores])
    data_min = np.min(scores)
    data_max = np.max(scores)
    scores = (scores - data_min) / (data_max - data_min)

    labels = np.hstack([pos_label_list, np.zeros(len(neg_scores))])
    data_min = np.min(labels)
    data_max = np.max(labels)
    labels = (labels - data_min) / (data_max - data_min)

    mse_value = metrics.mean_squared_error(labels, scores)

    return mse_value

def run_dataset(oriGraph_filename, train_filename, test_task):
    # num_nodes, raw_features, _, _, neigh_dict = load_cora(oriGraph_filename, directed=False)
    # num_nodes, raw_features, _, _, neigh_dict = load_cora()

    num_nodes, raw_features, neigh_dict, test_pos, test_neg, original_graph, trainGraph = load_dataset(oriGraph_filename, train_filename, test_task, directed=False)

    all_nodes = np.random.permutation(num_nodes)  # Added by sen

    if ENABLE_UNKNOWN_OP:
        # /graphsage/unsupervised_train.py, line 139
        raw_features = np.vstack([raw_features, np.zeros((raw_features.shape[1],))])

    minibatch_generator = generate_training_minibatch(trainGraph, neigh_dict
                                                      ,BATCH_SIZE
                                                      ,SAMPLE_SIZES
                                                      ,NEG_SIZE)

    graphsage = GraphSage(raw_features, INTERNAL_DIM, len(SAMPLE_SIZES), NEG_WEIGHT)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # training
    times = []

    func = lambda x: rdp_bank.RDP_gaussian({'sigma': args.sigma}, x)
    
    step = 0
    mean = 0.0  # Mean of the normal distribution
    stddev = 1.0  # Standard deviation of the normal distribution
    # Initialize embeddings with values sampled from a normal distribution
    all_embeddings = tf.random.normal((num_nodes, INTERNAL_DIM), mean=mean, stddev=stddev, dtype=tf.float32)

    for minibatch in islice(minibatch_generator, 0, TRAINING_STEPS):
        start_time = time.time()
        
        # privacy accoutant
        acct = rdp_acct.anaRDPacct()
    
        with tf.GradientTape() as tape:
            batch_embeddings = graphsage(minibatch)
            loss = graphsage.losses[0]

        sampled_nodes_index = minibatch.batch_all
        # use scatter to update all_embeddings
        all_embeddings = tf.tensor_scatter_nd_update(
            all_embeddings,
            tf.expand_dims(sampled_nodes_index, 1),
            batch_embeddings
        )

        grads = tape.gradient(loss, graphsage.trainable_weights)
        optimizer.apply_gradients(zip(grads, graphsage.trainable_weights))
        end_time = time.time()
        times.append(end_time - start_time)

        print('Indep_run_time', indep_run_time, 'Epoch', step)

        # --------- RDP mechanism -------------
        if args.RDP:
            sampling_prob = args.batch_size / trainGraph.number_of_edges()
            step = step + 1
            acct.compose_poisson_subsampled_mechanisms(func, sampling_prob, step)
            eps_now = acct.get_eps(args.delta)

            # Check if the current epsilon exceeds the allowed epsilon and tolerance threshold
            if eps_now > args.epsilon and (eps_now - args.epsilon) > args.eps_tolerance:
                print('jump out')
                break

    return all_embeddings, test_pos, test_neg, original_graph

if __name__ == "__main__":
    test_task = 'lwp'
    set_algo_name = 'PrivGNN'

    if test_task == 'lwp':
        # 'Reality-call', 'Digg-reply', 'Enron', 'Sub_wiki-talk0.2'
        set_dataset_names = ['Reality-call', 'Digg-reply', 'Enron', 'Sub_wiki-talk0.2']

        for set_dataset_name in set_dataset_names:
            set_eps_values = [6]

            for each_eps_value in set_eps_values:
                args.epsilon = each_eps_value

                for indep_run_time in range(args.indep_run_times):

                    tf.compat.v1.reset_default_graph()

                    set_split_name = 'train0.8_test0.2'
                    oriGraph_filename = './data/' + set_dataset_name + '/train_1'
                    train_filename = './data/' + set_dataset_name + '/' + set_split_name + '/'

                    start_time = time.time()

                    emb, test_pos, test_neg, original_graph = run_dataset(oriGraph_filename, train_filename, test_task)

                    embedding_mat = emb.numpy()

                    mse_value = CalcMSE2(original_graph, embedding_mat, test_pos, test_neg)

                    print('MSE_Value', mse_value)

    if test_task == 'StrucEqu':
        # 'w_Reality-call', 'w_Digg-reply', 'w_Enron', 'w_Sub_wiki-talk0.2'
        set_dataset_names = ['w_Reality-call', 'w_Digg-reply', 'w_Enron']

        for set_dataset_name in set_dataset_names:
            set_eps_values = [6]

            for each_eps_value in set_eps_values:
                args.epsilon = each_eps_value

                for indep_run_time in range(args.indep_run_times):

                    tf.compat.v1.reset_default_graph()

                    oriGraph_filename = './data/' + set_dataset_name + '/train_1'

                    start_time = time.time()

                    emb, _, _, original_graph = run_dataset(oriGraph_filename, oriGraph_filename, test_task)

                    # A = nx.to_numpy_matrix(original_graph)
                    # A = np.array(A)

                    A = nx.to_scipy_sparse_matrix(original_graph, format='csr')

                    pearson_vals = functions.structural_equivalence1(A, emb)
                    pearson_val = pearson_vals[0]

                    print('pearson_val', pearson_val)









