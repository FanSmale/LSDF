import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.myutils import get_nonlinear, init_random_seed


class MLP(nn.Module):
    def __init__(self, nodes_info, nonlinear_info, dropout_rate, device):
        super(MLP, self).__init__()

        num_layers = len(nodes_info) - 1
        self.FCLayers = nn.ModuleList()
        for i in range(num_layers):
            self.FCLayers.append(nn.Linear(nodes_info[i], nodes_info[i + 1]))
            nonlinear = get_nonlinear(nonlinear_info[i])
            if nonlinear is not None:
                self.FCLayers.append(nonlinear)
            self.FCLayers.append(nn.Dropout(p=dropout_rate))

        self.reset_parameters()
        self.to(device)

    def forward(self, input):
        for layer in self.FCLayers:
            input = layer(input)
        return input

    def reset_parameters(self):
        for layer in self.FCLayers:
            if layer.__class__.__name__ == 'Linear':
                nn.init.kaiming_uniform_(layer.weight, a=0.1, nonlinearity="leaky_relu")
                nn.init.uniform_(layer.bias, 0, 0.1)
            elif layer.__class__.__name__ == 'BatchNorm1d':
                layer.reset_parameters()


class Attention(nn.Module):
    def __init__(self, nodes_info, nonlinear_info, dropout_rate, device):
        super(Attention, self).__init__()
        self.NN_Q = nn.Linear(nodes_info[0], nodes_info[1])
        self.NN_K = nn.Linear(nodes_info[0], nodes_info[1])
        self.NN_V = nn.Linear(nodes_info[0], nodes_info[1])
        self.mlp = MLP(nodes_info, nonlinear_info, dropout_rate, device)

        self.reset_parameters()

    def forward(self, input):
        Q = self.NN_Q(input)                    # n × d1 × d2
        K = self.NN_K(input)                    # n × d1 × d2
        V = self.NN_V(input)                    # n × d1 × d2

        Q, K = F.normalize(Q, p=2, dim=2), F.normalize(K, p=2, dim=2)
        sim = torch.bmm(Q, K.transpose(1, 2))   # n × d1 × d1
        sim = torch.softmax(sim, dim=2)
        output = torch.bmm(sim, V)              # n × d1 × d2
        res = self.mlp(output)                  # n × d1 × d2

        return res

    def reset_parameters(self):
        self.NN_Q.reset_parameters()
        self.NN_K.reset_parameters()
        self.NN_V.reset_parameters()
        self.mlp.reset_parameters()


class GINLayer(nn.Module):
    def __init__(self, para_mlp, para_eps=0.0, para_train_eps=True):
        super(GINLayer, self).__init__()

        self.mlp = para_mlp
        self.initial_eps = para_eps

        if para_train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([para_eps]))
        else:
            self.register_buffer("eps", torch.Tensor([para_eps]))

        self.reset_parameters()

    def forward(self, para_input, para_adj):
        res = para_input
        n = len(para_input)

        # Aggregating neighborhood information
        neighs = torch.bmm(para_adj.unsqueeze(0).expand(n, -1, -1), res)

        # Reweighing the center node representation
        res = (1 + self.eps) * res + neighs

        # Updating node representations
        res = self.mlp(res)

        # Residual connection
        output = res + para_input if res.shape == para_input.shape else res

        return output

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.initial_eps)


class GIN(nn.Module):
    def __init__(self, para_nodes_info, para_nonlinear_info, para_dropout_rate, para_device, para_eps=0.0, para_train_eps=True):
        super(GIN, self).__init__()

        num_layers = len(para_nodes_info)
        self.GINLayers = nn.ModuleList()

        for i in range(num_layers):
            self.GINLayers.append(GINLayer(MLP(para_nodes_info[i], para_nonlinear_info[i], para_dropout_rate, para_device),
                                           para_eps, para_train_eps))

        self.reset_parameters()
        self.to(para_device)

    def forward(self, input, para_adj):
        for layer in self.GINLayers:
            input = layer(input, para_adj)

        return input

    def reset_parameters(self):
        for layer in self.GINLayers:
            layer.reset_parameters()


class LSDFNet(nn.Module):
    def __init__(self, para_feature_dim, para_label_dim, para_latent_dim, para_shape, para_dropout_rate, para_device):
        super(LSDFNet, self).__init__()
        self.label_dim = para_label_dim
        self.latent_dim = para_latent_dim
        self.shape = para_shape

        self.NN1 = MLP([para_feature_dim, para_shape[0] * para_shape[1]], ["leaky_relu"], para_dropout_rate, para_device)
        self.NN2 = Attention([para_shape[1], para_shape[1]], ["leaky_relu"], para_dropout_rate, para_device)
        self.NN3 = MLP([para_shape[0] * para_shape[1], para_latent_dim[0], para_latent_dim[1] * para_label_dim],
                       ["leaky_relu", "leaky_relu"], para_dropout_rate, para_device)
        self.NN4 = GIN([[para_latent_dim[1], para_latent_dim[1]]], [["leaky_relu"]], para_dropout_rate, para_device)
        self.cls_conv = nn.Conv1d(para_label_dim, para_label_dim, para_latent_dim[1], groups=para_label_dim)

        self.reset_parameters()
        self.to(para_device)

    def forward(self, input, para_adj):
        n = len(input)
        input = self.NN1(input)                                                         # n × d1d2
        input = torch.reshape(input, (n, self.shape[0], self.shape[1]))                 # n × d1 × d2
        input = self.NN2(input)                                                         # n × d1 × d2
        input = torch.reshape(input, (n, self.shape[0] * self.shape[1]))                # n × d1d2
        input = self.NN3(input)                                                         # n × qd
        input = torch.reshape(input, (n, self.label_dim, self.latent_dim[1]))           # n × q × d
        input = self.NN4(input, para_adj)                                                    # n × q × d
        scores = self.cls_conv(input).squeeze(2)

        return scores

    def reset_parameters(self):
        init_random_seed()
        self.NN1.reset_parameters()
        self.NN2.reset_parameters()
        self.NN3.reset_parameters()
        self.NN4.reset_parameters()
        self.cls_conv.reset_parameters()
