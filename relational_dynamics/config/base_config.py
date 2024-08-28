import numpy as np

import torch
import os

class BaseConfig(object):
    def __init__(self, args, dtype=torch.FloatTensor, **kwargs):
        self.args = args 
        self.result_dir = args.result_dir
        self.dtype = dtype

    def get_device(self):
        if self.dtype == torch.FloatTensor:
            return torch.device("cpu")
        else:
            return torch.device("cuda")

    def get_logger_dir(self):
        return os.path.join(self.result_dir, 'logs')

    def get_test_logger_dir(self):
        return os.path.join(self.result_dir, 'test_logs')

    def get_model_checkpoint_dir(self):
        return os.path.join(self.result_dir, 'checkpoint')