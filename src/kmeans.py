import random
import numpy as np

class myKMeans():

    def __init__(self, n_clusters, max_iter, limit):
        '''
        limit is the smallest distance within which we stop iteration
        '''
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.limit = limit

    def fit_predict(self, X):
        # find a random start point
        init_index = [random.randint(0, len(X) - 1) for i in range(self.n_clusters)]
        X = X.to_numpy()
        last_centroid = X[init_index]
        converge = False
        partition_index = None
        iters = 0
        # iteratively update centroid
        while not converge and iters < self.max_iter:
            distance = []
            for i in range(self.n_clusters):
                temp = X - last_centroid[i]
                distance.append(np.linalg.norm(temp, ord=2, axis=1))
            distance = np.stack(distance)
            # find the smallest
            partition_index = distance.argmin(0)
            partition = [X[partition_index == i] for i in range(self.n_clusters)]
            new_centroid = np.stack([partition[i].mean(0) for i in range(self.n_clusters)])
            if np.linalg.norm((new_centroid - last_centroid) * self.limit, ord=2, axis=1).mean() < 1:
                converge = True
            # update new centroid
            last_centroid = new_centroid
            iters += 1
        return partition_index