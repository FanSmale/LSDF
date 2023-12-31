import os
import json
import torch


class Properties:
    def __init__(self, para_file_name):
        # check whether the JSON file exists
        config_name = "./config/config.json"
        assert os.path.exists(config_name), "Config file is not accessible."

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # open json
        with open(config_name) as f:
            cfg = json.load(f)['lsdf']

        # assign the common parameters
        self.batch_size = cfg["common"]["batch_size"]
        self.checkpoint_path = cfg["common"]["checkpoint_path"]
        self.lr = cfg["common"]["lr"]
        self.weight_decay = cfg["common"]["weight_decay"]

        # check whether the information of this mat exists in the config file
        assert para_file_name in cfg.keys(), "".join(
            ["The parameters of ", para_file_name, " are not defined in the JSON file of config."])

        # read parameters from the mat file
        self.is_text_data = cfg[para_file_name]["is_text_data"]
        self.dropout_rate = cfg[para_file_name]["dropout_rate"]
        self.filename = cfg[para_file_name]["filename"]
        self.feature_dim = None
        self.label_dim = None
        self.latent_dim = cfg[para_file_name]["latent_dim"]
        self.max_epoch = cfg[para_file_name]["max_epoch"]
        self.shape = cfg[para_file_name]["shape"]
