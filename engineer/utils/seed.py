import torch
import torch.backends.cudnn
import os
import random
import numpy as np
import torch_geometric


def set_seed(seed: int):
    # seed = int(seed)
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    # torch_geometric.seed.seed_everything(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # torch.use_deterministic_algorithms(True)
    # os.environ["PL_GLOBAL_SEED"] = str(seed)
    # torch.set_float32_matmul_precision("high")

    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
