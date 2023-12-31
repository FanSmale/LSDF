import os
import torch

from loss.loss import ClassificationLoss
from model.network import LSDFNet
from optimizer.optimization import Optimization


class LSDFModel:
    def __init__(self, prop):
        # device
        self.device = prop.device

        # model
        self.model = LSDFNet(prop.feature_dim, prop.label_dim, prop.latent_dim, prop.shape, prop.dropout_rate, prop.device)

        # loss
        cls_loss = ClassificationLoss().to(prop.device)
        self.criterion = [cls_loss]

        # optimizer
        self.optimizer = Optimization(prop.max_epoch, prop.batch_size, prop.checkpoint_path, prop.filename)
        self.lr = prop.lr
        self.weight_decay = prop.weight_decay

        # dataset information
        self.checkpoint_path = prop.checkpoint_path
        self.filename = prop.filename
        self.label_adj = None

        self.reset_parameters()

    def reset_parameters(self):
        self.model.reset_parameters()

    def train(self, para_train, para_val):
        # transform data to tensor and compute the label adjacent matrix
        train_data = torch.tensor(para_train[0], dtype=torch.float32).to(self.device)
        train_target = torch.tensor(para_train[1], dtype=torch.float32).to(self.device)
        val_data = torch.tensor(para_val[0], dtype=torch.float32).to(self.device)
        val_target = torch.tensor(para_val[1], dtype=torch.float32).to(self.device)
        self.label_adj = self.acquire_adj(train_target).to(self.device)

        # learn
        self.optimizer.learn(self.model, self.criterion, train_data, train_target, val_data, val_target, self.label_adj,
                             self.lr, self.weight_decay)

    def predict(self, para_test_data):
        self.model.eval()
        self.load_checkpoint()
        with torch.no_grad():
            test_data = torch.tensor(para_test_data, dtype=torch.float32).to(self.device)
            output = self.model(test_data, self.label_adj)

        return output.sigmoid_().cpu().detach().numpy()

    def load_checkpoint(self):
        checkpoint = os.path.join(self.checkpoint_path, self.filename + ".pth")
        if os.path.isfile(checkpoint):
            checkpoint = torch.load(checkpoint)
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            print('No checkpoint is found at {}.'.format(checkpoint))

    @staticmethod
    def acquire_adj(para_labels):
        # compute label correlation matrix
        adj = torch.matmul(para_labels.t(), para_labels)
        y_sum = torch.sum(para_labels.t(), dim=1, keepdim=True)
        y_sum[y_sum < 1e-6] = 1e-6
        adj = adj / y_sum
        adj = (adj + adj.t()) * 0.5

        # keep self-loop
        q = adj.size(0)
        res = torch.zeros(adj.size())
        for i in range(q):
            res[i, i] = adj[i, i]
            adj[i, i] = 0

        # three-order correlation: just keep the most relevant and least relevant labels
        max_indices, min_indices = torch.argmax(adj, dim=1), torch.argmin(adj, dim=1)
        for i in range(q):
            if max_indices[i] != i:
                res[i, max_indices[i]] = adj[i, max_indices[i]]
            if min_indices[i] != i:
                res[i, min_indices[i]] = adj[i, min_indices[i]]

        return res
