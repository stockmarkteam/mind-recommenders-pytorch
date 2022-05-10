import random

import numpy as np
import torch
from omegaconf import OmegaConf


def init_random_seed(seed=0):
    # cf. https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_omegaconf_resolvers():
    # line_count:
    OmegaConf.register_new_resolver("line_count", lambda fname: None if fname is None else sum([1 for _ in open(fname)]))
