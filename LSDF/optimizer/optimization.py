import math
import os
import torch


class Optimization:
    def __init__(self, para_max_epoch, para_batch_size, para_checkpoint_path, para_filename):
        self.max_epoch = para_max_epoch
        self.batch_size = para_batch_size
        self.checkpoint_path = para_checkpoint_path
        self.filename = para_filename

    def learn(self, para_model, para_criterion, para_train_data, para_train_target, para_val_data, para_val_target,
              para_label_adj, para_lr=1e-3, para_weight_decay=1e-4):
        optimizer = torch.optim.Adam(para_model.parameters(), lr=para_lr, weight_decay=para_weight_decay)

        best_loss = torch.tensor(float("inf"))
        for i in range(self.max_epoch):
            train_loss, val_loss = 0, 0

            # train: if batch is lager than given size, we will split data into small batch
            if self.batch_size >= para_train_data.size(0):
                train_loss = self.train(para_model, para_criterion, para_train_data, para_train_target, para_label_adj,
                                        optimizer)
            else:
                num_data = para_train_data.size(0)
                index = torch.randperm(num_data)
                max_iters = math.ceil(num_data / self.batch_size)
                for i in range(max_iters):
                    low = i * self.batch_size
                    high = min(low + self.batch_size, num_data)
                    batch_train_data = para_train_data[index[low:high], :]
                    batch_train_target = para_train_target[index[low:high], :]
                    train_loss += self.train(para_model, para_criterion, batch_train_data, batch_train_target,
                                             para_label_adj, optimizer)

            # validate: if batch is lager than given size, we will split data into small batch
            if self.batch_size >= para_val_data.size(0):
                val_loss = self.validate(para_model, para_criterion, para_val_data, para_val_target, para_label_adj)
            else:
                num_data = para_val_data.size(0)
                index = torch.randperm(num_data)
                max_iters = math.ceil(num_data / self.batch_size)
                for i in range(max_iters):
                    low = i * self.batch_size
                    high = min(low + self.batch_size, num_data)
                    batch_val_data = para_val_data[index[low:high], :]
                    batch_val_target = para_val_target[index[low:high], :]
                    val_loss += self.train(para_model, para_criterion, batch_val_data, batch_val_target, para_label_adj,
                                           optimizer)

            # save checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint({"state_dict": para_model.state_dict()})

    def train(self, para_model, para_criterion, para_train_data, para_train_target, para_label_adj, para_optimizer):
        para_model.train()
        predict = para_model(para_train_data, para_label_adj)

        cls_loss = para_criterion[0](predict, para_train_target)
        loss = cls_loss

        para_optimizer.zero_grad()
        loss.backward()
        para_optimizer.step()

        return cls_loss.data.item()

    def validate(self, para_model, para_criterion, para_val_data, para_val_target, para_label_adj):
        para_model.eval()
        with torch.no_grad():
            predict = para_model(para_val_data, para_label_adj)
            cls_loss = para_criterion[0](predict, para_val_target)
            loss = cls_loss

        return loss.data.item()

    def save_checkpoint(self, para_checkpoint):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        torch.save(para_checkpoint, os.path.join(self.checkpoint_path, self.filename + ".pth"))
