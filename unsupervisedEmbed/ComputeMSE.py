import numpy as np
from sklearn import metrics

def CalcMSE(graph, W_emb, test_pos, test_neg):
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
