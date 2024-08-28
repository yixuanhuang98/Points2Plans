import numpy as np
import argparse
import pickle
import sys
import os
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import copy
import random
import torch.nn.functional as F

sys.path.append(os.getcwd())
from relational_dynamics.utils.colors import bcolors
from relational_dynamics.utils.other_util import create_log_dirs
from relational_dynamics.utils import parse_util
from relational_dynamics.config.base_config import BaseConfig
from relational_dynamics.base_RD import RelationalDynamics

def main(args):
    dtype = torch.FloatTensor
    if args.cuda:
        dtype = torch.cuda.FloatTensor

    config = BaseConfig(args, dtype=dtype)
    create_log_dirs(config)

    if len(args.checkpoint_path) > 0:
        planner = RelationalDynamics(config)
        planner.load_checkpoint(args.checkpoint_path)
        result_dict = planner.RD(train=False, threshold = 0)
    else:
        trainer = RelationalDynamics(config)
        result_dict = trainer.RD()
        
if __name__ == '__main__':
    parser = parse_util.get_parser()
    args = parser.parse_args()
    np.set_printoptions(precision=4, linewidth=120)
    if args.set_random_seed:
        seed = args.seed  
        random.seed(args.seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main(args)
