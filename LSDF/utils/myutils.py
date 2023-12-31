import os
import random
import torch
import torch.nn as nn
import scipy.io as scio

from sklearn import preprocessing
from sklearn.decomposition import PCA

from utils.metric import *
from utils.properties import Properties


class ResultHandling:
    """ compute, show and save the results """
    def __init__(self, para_filename, para_metric_scores):
        self.filename = para_filename
        self.metric_scores = para_metric_scores          # shape is (5, 10), 5 cv and 10 evaluation metrics
        self.metric_list = ["acc", "ap", "cov", "hl", "macro_auc", "micro_auc", "ndcg", "oe", "peak-f1", "rkl"]
        self.results_path = "./results"

    def compute_results(self):
        metric_scores = np.array(self.metric_scores).transpose()
        means = np.mean(metric_scores, axis=1)
        stds = np.std(metric_scores, axis=1)

        return means, stds

    def show(self):
        means, stds = self.compute_results()
        print("=====================", self.filename, "=====================")
        for i, metric_name in enumerate(self.metric_list):
            print(metric_name, f"{means[i]:.4f}", "±", f"{stds[i]:.4f}")

    def save_results(self):
        means, stds = self.compute_results()
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        file_path = os.path.join(self.results_path, self.filename + ".txt")

        file = open(file_path, "w+")
        file.write("acc: {:.4f}±{:.4f}\n".format(means[0], stds[0]))
        file.write("ap: {:.4f}±{:.4f}\n".format(means[1], stds[1]))
        file.write("cov: {:.4f}±{:.4f}\n".format(means[2], stds[2]))
        file.write("hl: {:.4f}±{:.4f}\n".format(means[3], stds[3]))
        file.write("macro auc: {:.4f}±{:.4f}\n".format(means[4], stds[4]))
        file.write("micro auc: {:.4f}±{:.4f}\n".format(means[5], stds[5]))
        file.write("ndcg: {:.4f}±{:.4f}\n".format(means[6], stds[6]))
        file.write("oe: {:.4f}±{:.4f}\n".format(means[7], stds[7]))
        file.write("pf1: {:.4f}±{:.4f}\n".format(means[8], stds[8]))
        file.write("rkl: {:.4f}±{:.4f}\n".format(means[9], stds[9]))
        file.close()


def compute_metrics(para_predict, para_target):
    res = [accuracy(para_target, para_predict), average_precision(para_target, para_predict), coverage(para_target, para_predict),
           hamming_loss(para_target, para_predict), macro_averaging_auc(para_target, para_predict), micro_averaging_auc(para_target, para_predict),
           ndcg(para_target, para_predict), one_error(para_target, para_predict), peak_f1_score(para_target, para_predict),
           ranking_loss(para_target, para_predict)]
    res = np.array(res)
    
    return res


def init_random_seed(para_seed=0):
    torch.manual_seed(para_seed)
    torch.cuda.manual_seed(para_seed)
    torch.cuda.manual_seed_all(para_seed)
    np.random.seed(para_seed)
    random.seed(para_seed)


def get_nonlinear(para_nonlinear, para_negative_slope=0.1):
    nonlinear_map = {"leaky_relu": nn.LeakyReLU(para_negative_slope, inplace=True),
                     "relu": nn.ReLU(inplace=True),
                     "softmax": nn.Softmax(dim=1),
                     "sigmoid": nn.Sigmoid(),
                     "tanh": nn.Tanh()}

    return nonlinear_map.get(para_nonlinear)


def load(para_path, para_dataset, para_reduction_dim_ratio=0.1):
    file_path = os.path.join(para_path, para_dataset + ".mat")
    prop, file = Properties(para_dataset), scio.loadmat(file_path)

    data = np.concatenate((file["train_data"], file["test_data"]))
    target = np.concatenate((file["train_target"].transpose(), file["test_target"].transpose()))

    data = preprocessing.scale(data)
    target[target == -1] = 0

    if prop.is_text_data:
        pca = PCA(n_components=int(len(data[0]) * para_reduction_dim_ratio), random_state=0)
        data = pca.fit_transform(data)

    prop.feature_dim, prop.label_dim = len(data[0]), len(target[0])

    return prop, data, target
