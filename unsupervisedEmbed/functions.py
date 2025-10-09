import random
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

def embedding_distance(embedding, metric='euclidean'):
    """Returns a 2D-array whose [i][j] is the distance between vector i and j of embedding """
    return cdist(embedding, embedding, metric)

def structural_equivalence1(adjacency_matrix, embedding, metric='euclidean'):
    '''
    Compute structural equivalence: Correlation between embedding space distance and structural distance.

    Parameters:
        adjacency_matrix: Adjacency matrix (can be sparse)
        embedding: Node embedding vectors
        metric: Distance metric method

    Returns:
        Pearson correlation coefficient and p-value
    '''

    # For sparse matrices, use shape[0] to get the number of nodes, compatible with dense matrices
    N = adjacency_matrix.shape[0] # Fix: use shape[0] instead of len()

    # Sample nodes to avoid memory issues with too many nodes
    if N > 2000:
        sample_ratio = 0.1  # 0.01 for wiki-talk0.2, 0.1 for other datasets
        sample_size = int(N * sample_ratio)
        sampled_nodes = random.sample(range(N), sample_size)
    else:
        sampled_nodes = list(range(N))   # Use all nodes for smaller graphs

    # Extract embeddings and adjacency vectors for sampled nodes
    sampled_emb = [embedding[node] for node in sampled_nodes]

    # For sparse matrices, convert to array for computation, or use .toarray()[node] to get a row
    sampled_adj = [adjacency_matrix[node].toarray().flatten() for node in sampled_nodes]

    sample_size = len(sampled_nodes)

    # Compute distance matrices
    emb_dist = cdist(sampled_emb, sampled_emb, metric=metric)
    struc_dist = cdist(sampled_adj, sampled_adj, metric=metric)

    # Extract upper triangular distance pairs
    struc_pairs = (struc_dist[i, j] for i in range(sample_size)
                   for j in range(i + 1, sample_size))
    emb_pairs = (emb_dist[i, j] for i in range(sample_size)
                 for j in range(i + 1, sample_size))

    # Compute and return the Pearson correlation coefficient
    return pearsonr(list(struc_pairs), list(emb_pairs))

def structural_equivalence(adjacency_matrix, embedding, metric='euclidean'):
    '''
    metric : str or callable, optional
    The distance metric to use.  If a string, the distance function can be
    'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
    'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
    'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
    'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
    'wminkowski', 'yule'.
    '''
    N = len(adjacency_matrix) #number of nodes
    if N > 2000: 
        A = [i for i in range(N)]
        fraction_of_sampled_nodes = 0.1
        number_of_sampled_nodes = int(N*fraction_of_sampled_nodes)
        list_of_random_nodes = random.sample(A, number_of_sampled_nodes)
        #if there are more than 2k nodes, then we sample 10% of them
    else:
        #else we take all the nodes
        list_of_random_nodes = range(N)
    
    less_emb = []
    less_adjM = []
    
    N2 = len(list_of_random_nodes)
    for w in list_of_random_nodes:
        less_emb.append(embedding[w])
        less_adjM.append(adjacency_matrix[w])

    # print('computing distances') #compute distances between vectors of the embedding space
    dist = embedding_distance(less_emb, metric)
    struc_dist = embedding_distance(less_adjM, metric) #compute distances between lines of the adj matrix
    struc_distances_list = []
    embedding_distances = []

    for i in range(N2):
        for j in range(N2):
            if i<j: #we don't take i=j since distances will be both 0, and d[i][j] = d[j][i]
                struc_distances_list.append(struc_dist[i][j])
                embedding_distances.append(dist[i][j])
    # print('Structural equivalence computed')
    # return spearmanr(struc_distances_list, embedding_distances)
    return pearsonr(struc_distances_list, embedding_distances) #return the correlation coefficient



