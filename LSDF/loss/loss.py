import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()

    def forward(self, para_predict_score, para_target):
        # classification loss
        bce = nn.BCELoss()
        cls_loss = bce(para_predict_score.sigmoid_().view(para_target.shape), para_target)

        return cls_loss
