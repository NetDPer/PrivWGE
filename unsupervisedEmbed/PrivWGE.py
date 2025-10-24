import tensorflow as tf
import numpy as np
import argparse
import networkx as nx
from sklearn.externals import joblib
import functions
from autodp import rdp_acct, rdp_bank
import ComputeMSE

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', default=512)
parser.add_argument('--batch_size', default=2000)
parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
parser.add_argument('--edge_sampling', default='poisson', help='numpy or atlas or uniform')
parser.add_argument('--node_sampling', default='numpy', help='numpy or atlas or uniform')
parser.add_argument('--lr', default=0.01)
parser.add_argument('--K', default=5)
parser.add_argument('--sigma', default=5)
parser.add_argument('--delta', default=10**(-5))
parser.add_argument('--epsilon', default=6)
parser.add_argument('--clip_value', default=0.5)
parser.add_argument('--RDP', default=True)
parser.add_argument('--shared_mat', default=True)
parser.add_argument('--random_seed', default=42)
parser.add_argument('--eps_tolerance', default=0.01)
parser.add_argument('--n_epoch', default=50)  # all datasets are set to 5
parser.add_argument('--indep_run_times', default=1)  # done

args = parser.parse_args()  # parameter

class PrivWGE_Model:
    def __init__(self):
        with tf.compat.v1.variable_scope('forward_pass'):
            tf.compat.v1.disable_eager_execution()
            self.u_i = tf.compat.v1.placeholder(name='u_i', dtype=tf.int32, shape=[None])
            self.u_j = tf.compat.v1.placeholder(name='u_j', dtype=tf.int32, shape=[None])
            self.label = tf.compat.v1.placeholder(name='label', dtype=tf.float32, shape=[None])
            self.w_ij = tf.compat.v1.placeholder(name='w_ij', dtype=tf.float32, shape=[None])

            self.shared_matrix = tf.compat.v1.get_variable('shared_w_mat', [args.embedding_dim, args.embedding_dim],
                                                           initializer=tf.random_uniform_initializer(minval=-1.,
                                                                                                     maxval=1.,
                                                                                                     seed=args.random_seed))

            self.embedding = tf.compat.v1.get_variable('target_emb', [args.num_of_nodes, args.embedding_dim],
                                                           initializer=tf.random_uniform_initializer(minval=-1.,
                                                                                                     maxval=1.,
                                                                                                     seed=args.random_seed))
            self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=args.num_of_nodes), self.embedding)

            if args.proximity == 'first-order':
                self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
            elif args.proximity == 'second-order':
                self.context_embedding = tf.compat.v1.get_variable('context_emb', [args.num_of_nodes, args.embedding_dim],
                                                           initializer=tf.random_uniform_initializer(minval=-1.,
                                                                                                     maxval=1.,
                                                                                                     seed=args.random_seed))
                self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.context_embedding)

            if args.shared_mat:
                # Ensure all values in shared_matrix are >= 0
                self.shared_matrix = tf.maximum(self.shared_matrix, 0)
                self.u_i_embedding = tf.maximum(self.u_i_embedding, 0)
                self.u_j_embedding = tf.maximum(self.u_j_embedding, 0)
                self.inner_product = tf.reduce_sum(tf.matmul(self.u_i_embedding, self.shared_matrix) * self.u_j_embedding, axis=1)
            else:
                self.u_i_embedding = tf.maximum(self.u_i_embedding, 0)
                self.u_j_embedding = tf.maximum(self.u_j_embedding, 0)
                self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)

            self.sgm_loss = -tf.compat.v1.log_sigmoid(self.label * self.inner_product) * self.w_ij
            self.loss = tf.reduce_mean(self.sgm_loss)

            self.optimizer = tf.compat.v1.train.AdamOptimizer(args.lr)
            self.params = [v for v in tf.compat.v1.trainable_variables() if 'forward_pass' in v.name]

            if args.RDP:
                self.var_list = self.params
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss, self.var_list)
                for i, (g, v) in enumerate(self.grads_and_vars):
                    if g is not None and v is not None:
                        g = tf.clip_by_norm(g, args.clip_value)
                        stddev = args.sigma * args.clip_value / args.batch_size
                        g = g + tf.compat.v1.random_normal(tf.shape(g), stddev=stddev)
                        self.grads_and_vars[i] = (g, v)
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)
            else:
                self.train_op = self.optimizer.minimize(self.loss)

class AliasSampling:
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int64)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res

class prepare_data:
    def __init__(self, graph_file=None):
        self.g = graph_file
        self.num_of_nodes = len(self.g.nodes())
        self.num_of_edges = len(self.g.edges())
        self.edges_raw = self.g.edges(data=True)
        self.nodes_raw = self.g.nodes(data=True)

        self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float64)
        self.edge_distribution /= np.sum(self.edge_distribution)
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)

        # self.node_negative_distribution = np.power(
        #     np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float64), 0.75)

        self.node_negative_distribution = np.array([1/self.num_of_nodes for node, _ in self.nodes_raw], dtype=np.float64)

        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        self.edges = [(u, v) for u, v, _ in self.edges_raw]

    def prepare_data(self):
        global edge_batch_index, negative_node

        if args.edge_sampling == 'numpy':
            edge_batch_index = np.random.choice(self.num_of_edges, size=args.batch_size, p=self.edge_distribution)
        elif args.edge_sampling == 'atlas':
            edge_batch_index = self.edge_sampling.sampling(args.batch_size)
        elif args.edge_sampling == 'uniform':
            edge_batch_index = np.random.randint(0, self.num_of_edges, size=args.batch_size)
        elif args.edge_sampling == 'poisson':
            prob = args.batch_size / self.num_of_edges
            edge_batch_index = poisson_subsample_indices(self.num_of_edges, q=prob)

        u_i = []
        u_j = []
        label = []
        w_ij = []

        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            if self.g.__class__ == nx.Graph:
                if np.random.rand() > 0.5:
                    edge = (edge[1], edge[0])

            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            w_ij.append(self.g.get_edge_data(edge[0], edge[1])['weight'])

            for i in range(args.K):
                while True:
                    if args.node_sampling == 'numpy':
                        negative_node = np.random.choice(self.num_of_nodes, p=self.node_negative_distribution)
                    elif args.node_sampling == 'atlas':
                        negative_node = self.node_sampling.sampling()
                    elif args.node_sampling == 'uniform':
                        negative_node = np.random.randint(0, self.num_of_nodes)
                    # if not self.g.has_edge(self.node_index_reversed[negative_node], self.node_index_reversed[edge[0]]):
                    if not self.g.has_edge(negative_node, edge[0]):
                        break

                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)
                w_ij.append(self.g.get_edge_data(edge[0], edge[1])['weight'])

        return u_i, u_j, label, w_ij

class trainModel:
    def __init__(self, inf_display, graph, original_graph=None, test_pos=None, test_neg=None):
        self.inf_display = inf_display
        self.test_pos = test_pos
        self.test_neg = test_neg
        self.graph = graph
        self.original_graph = original_graph
        self.data_loader = prepare_data(self.graph)
        args.num_of_nodes = self.data_loader.num_of_nodes
        args.num_of_edges = self.data_loader.num_of_edges
        self.model = PrivWGE_Model()

    def train(self, test_task=None):
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')

        func = lambda x: rdp_bank.RDP_gaussian({'sigma': args.sigma}, x)

        for indep_run_time in range(args.indep_run_times):
            found = False

            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())

                for each_epoch in range(args.n_epoch):
                    # privacy accoutant
                    acct = rdp_acct.anaRDPacct()
                    '''
                    The acct = rdp_acct.anaRDPacct() is called in every iteration of the loop, 
                    so it will initialize a new instance of anaRDPacct() every time the loop iterates.
                    '''
                
                    u_i, u_j, label, w_ij = self.data_loader.prepare_data()

                    feed_dict = {self.model.u_i: u_i, self.model.u_j: u_j, self.model.label: label,
                                 self.model.w_ij: w_ij}
                    _, loss = sess.run([self.model.train_op, self.model.loss], feed_dict=feed_dict)

                    if args.shared_mat:
                        W_emb = sess.run(tf.matmul(self.model.embedding, self.model.shared_matrix))
                    else:
                        W_emb = sess.run(self.model.embedding)

                    print('Indep_run_time', indep_run_time, 'Epoch', each_epoch)

                    # --------- RDP mechanism -------------
                    if args.RDP:
                        sampling_prob = args.batch_size / self.graph.number_of_edges()
                        steps = each_epoch + 1
                        acct.compose_poisson_subsampled_mechanisms(func, sampling_prob, steps)
                        eps_now = acct.get_eps(args.delta)
                            
                        # Check if the current epsilon exceeds the allowed epsilon and tolerance threshold
                        if eps_now > args.epsilon and (eps_now - args.epsilon) > args.eps_tolerance:
                            print('jump out')
                            found = True
                            break

                    if found:
                        break

                if test_task == 'lwp':
                    mse_value = ComputeMSE.CalcMSE(self.original_graph, W_emb, test_pos, test_neg)

                    print('Indep_run_time', indep_run_time, 'Epoch', each_epoch, 'MSE_Value', mse_value)

                if test_task == 'StrucEqu':
                    A = nx.to_scipy_sparse_matrix(trainGraph, format='csr')
                    pearson_vals = functions.structural_equivalence1(A, W_emb)
                    pearson_val = pearson_vals[0]

                    print('Indep_run_time', indep_run_time, 'Epoch', each_epoch, 'pearson_val', pearson_val)

def poisson_subsample_indices(N=None, q=None):
    """
    Poisson subsampling for privacy amplification in DP.
    Each sample is included independently with probability q.

    Args:
        N: total number of samples
        q: sampling probability (0 < q < 1)

    Returns:
        indices: numpy array of selected indices
    """
    mask = np.random.rand(N) < q
    indices = np.where(mask)[0]

    return indices

def loadGraphFromEdgeListTxt(file_name, directed=True):
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

    return G

if __name__ == '__main__':
    test_task = 'StrucEqu'  # lwp or StrucEqu
    set_algo_name = 'PrivWGE'

    if test_task == 'lwp':
        # 'Reality-call', 'Digg-reply', 'Enron', 'Sub_wiki-talk0.2'
        set_dataset_names = ['Reality-call', 'Digg-reply', 'Enron', 'Sub_wiki-talk0.2']

        set_split_name = 'train0.8_test0.2'
        set_nepoch_name = 'nepoch' + str(args.n_epoch)
        set_emb_dim = 'dim' + str(args.embedding_dim)
        set_learning_rate = 'step' + str(args.lr)
        set_clip_value = 'clip' + str(args.clip_value)

        for set_dataset_name in set_dataset_names:
            set_eps_values = [6]

            for each_eps_value in set_eps_values:
                args.epsilon = each_eps_value

                tf.compat.v1.reset_default_graph()
                oriGraph_filename = '../data/' + set_dataset_name + '/train_1'
                train_filename = '../data/' + set_dataset_name + '/' + set_split_name + '/'

                # Load graph
                trainGraph = loadGraphFromEdgeListTxt(oriGraph_filename, directed=False)

                original_graph = trainGraph

                trainGraph = nx.adjacency_matrix(trainGraph)

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
                trainGraph = nx.from_scipy_sparse_matrix(trainGraph)

                print('Num nodes: %d, num edges: %d' % (trainGraph.number_of_nodes(), trainGraph.number_of_edges()))
                inf_display = [test_task, set_dataset_name]
                tm = trainModel(inf_display, trainGraph, original_graph=original_graph, test_pos=test_pos, test_neg=test_neg)
                tm.train(test_task=test_task)

    if test_task == 'StrucEqu':
        # 'w_Reality-call', 'w_Digg-reply', 'w_Enron', 'w_Sub_wiki-talk0.2'
        set_dataset_names = ['w_Reality-call', 'w_Digg-reply', 'w_Enron']

        for set_dataset_name in set_dataset_names:
            set_eps_values = [6]
            for each_eps_value in set_eps_values:
                args.epsilon = each_eps_value

                tf.compat.v1.reset_default_graph()
                oriGraph_filename = '../data/' + set_dataset_name + '/train_1'

                # Load graph
                trainGraph = loadGraphFromEdgeListTxt(oriGraph_filename, directed=False)

                original_graph = trainGraph

                print('Num nodes: %d, num edges: %d' % (trainGraph.number_of_nodes(), trainGraph.number_of_edges()))
                inf_display = [test_task, set_dataset_name]
                tm = trainModel(inf_display, trainGraph, original_graph=original_graph)
                tm.train(test_task=test_task)



