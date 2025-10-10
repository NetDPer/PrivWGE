### Requirements

- python 3.6.5

- Tensorflow 2.2.0

- numpy 1.19.5

- networkx 2.2

### Run PrivWGE

- MSE metric for link weight prediction is in ComputeMSE.py

- PCC (Pearson correlation coefficient) metric for weighted structural equivalence is in functions.py 

- run PrivWGE.py in unsupervisedEmbed

### Run PrivGNN

Note that PrivGNN uses cross-entropy as the loss function and achieves unsupervised learning through random edge sampling. Additionally, PrivGNN incorporates the perturbation technique from the classical GAP method to inject Gaussian noise into the aggregation process during each iteration. For PrivGNN, since it does not generate a node embedding matrix that matches the number of nodes in the graph during optimization, we predefine a matrix of the same size as the node set and incrementally update its entries in each iteration.

- run unsupervised.py in PrivGNN file for link weight prediction and weighted structural equivalence

  
