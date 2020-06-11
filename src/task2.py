#!/usr/bin/env python
# coding: utf-8

# basic libs
import pandas as pd
import sklearn as sk
import numpy as np
import argparse
import time
# built in clustering methods
from sklearn.cluster import SpectralClustering, KMeans
# self implemented clustering methods
from kmeans import myKMeans
# data/pipeline related operations
from sklearn.preprocessing import StandardScaler
# evaluation metrics
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
# visualization
import seaborn as sns
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='A clustering task for frogs MFCC')

parser.add_argument('--drop_feature_list', nargs='+', default=[], \
    help="a list of features that will not be used in training.")
parser.add_argument('--algorithm', type=str, default="KMeans", choices=["kmeans", "spectral", "mykmeans"],\
    help='clustering algorithm to be used.')
parser.add_argument('--max_iter', type=int, default=1000, metavar='N',
    help='max number of iters to perform when training.')
parser.add_argument('--class_number', type=int, default=4, choices=[4,8,10],\
    help="number of classes. Family: 4, Genus: 8, Species: 10")
parser.add_argument('--visualize', action='store_true', help='do visualization')

args = parser.parse_args()

# load the dataset, shuffle it
mfcc = pd.read_csv('./data/clustering/Frogs_MFCCs.csv', sep=',',header=0).sample(frac=1)

# select features
columns = list(mfcc.columns)
selected_columns = [name for name in columns if name not in args.drop_feature_list and name.startswith("MFCC")]
y = mfcc["Species"]
X = mfcc[selected_columns]
print("\nThe X data field names:")
print(selected_columns)


if args.algorithm == "kmeans":
    algorithm = KMeans(n_clusters=args.class_number, random_state=0, max_iter=args.max_iter)
elif args.algorithm == "spectral":
    algorithm = SpectralClustering(n_clusters=args.class_number, assign_labels="discretize", random_state=0, gamma=1.5)
elif args.algorithm == "mykmeans":
    algorithm = myKMeans(n_clusters=args.class_number, max_iter=args.max_iter, limit=100000)


# train
print("\nstart training with {} model...".format(args.algorithm))
time_start = time.time()
predicted = algorithm.fit_predict(X)
time_end = time.time()

similarity = adjusted_rand_score(y, predicted)
silhouette = silhouette_score(X, predicted, metric='euclidean')
time_elapsed = time_end - time_start

print("\ntraining finished in {:.3f} seconds".format(time_elapsed))
print("The similarity score between predicted and groundtruth is {:.3f}.".format(similarity))
print("The silhouette score is {:.3f}.".format(silhouette))

if args.visualize:
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    mfcc['y'] = predicted
    X_embedded = tsne.fit_transform(X)
    mfcc['tsne-2d-one'] = X_embedded[:,0]
    mfcc['tsne-2d-two'] = X_embedded[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", args.class_number),
        data=mfcc,
        legend="full",
        alpha=0.3
    )
    filename = 'visualization_class{}.png'.format(args.class_number)
    plt.savefig(filename)
    print("visualization file saved at {}".format(filename))
