#!/usr/bin/env python
# coding: utf-8

# basic libs
import pandas as pd
import sklearn as sk
import numpy as np
import argparse
import time
# built in classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
# self-implemented logistic regression
from logistic_regression import MyLogisticRegression
# data/pipeline related operations
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
# evaluation / hyper parameter tools
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

parser = argparse.ArgumentParser(description='A classification task for bank costumer prediction')

parser.add_argument('--drop_feature_list', nargs='+', default=[], \
    help="a list of features that will not be used in training.")
parser.add_argument('--classifier', type=str, default="logistic", choices=["logistic", "random_forest", "mlp", "svm", "mylogistic"],\
    help='classifier to be used.')
parser.add_argument('--folds', type=int, default=4, metavar='N',
    help='number of folds, default set to 5.')
parser.add_argument('--max_iter', type=int, default=1000, metavar='N',
    help='max number of iters to perform when training.')

args = parser.parse_args()

# load the dataset, shuffle it, prepare for k fold evaluation
bank_data = pd.read_csv('./data/classification/train_set.csv', sep=',',header=0).sample(frac=1)


# split the data into X and Y
X = bank_data.iloc[:, 1:-1]
y = bank_data.iloc[:, -1]
X_head = X.columns
dims = len(X_head)
print("\nThe X data field names:")
print([name for name in list(X_head) if name not in args.drop_feature_list])

kf = KFold(n_splits=args.folds)
kf.get_n_splits(X)

# data preprocessing, assemble the pipeline

numeric_features = []
categorical_features = []
# split features by datatype
for head, value in X.iteritems():
    if not head in args.drop_feature_list:
        if value.dtype == np.int64:
            numeric_features.append(head)
        else:
            categorical_features.append(head)

# transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

if args.classifier == "logistic":
    classifier = LogisticRegression(max_iter=args.max_iter)
elif args.classifier == "random_forest":
    classifier = RandomForestClassifier()
elif args.classifier == "mlp":
    classifier = MLPClassifier(max_iter=args.max_iter)
elif args.classifier == "svm":
    classifier = svm.SVC(max_iter=args.max_iter)
else:
    classifier = MyLogisticRegression(0.001, max_iter=args.max_iter, batch_size=512, lambda_normalization=0.01)

# pipeline 
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', classifier)])


# train and test
print("\nstart training with {} model...".format(args.classifier))
time_start = time.time()

predicted = cross_val_predict(clf, X, y, cv=kf, n_jobs=1)
report = classification_report(y, predicted)
print(predicted.mean())
time_end = time.time()
time_elapsed = time_end - time_start

print("\ntraining finished in {:.3f} seconds".format(time_elapsed))
print("\nevaluation setting\nusing K-Fold evaluation, where K is set to: {}\n".format(args.folds))
print(report)
