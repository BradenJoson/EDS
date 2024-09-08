import torch
import torch.nn as nn
import random
import numpy as np
import os
from configs.settings import get_settings


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
cfgs = get_settings()
set_random_seed(seed=cfgs.seed)


from train import train_fn
from test import test_fn
from utils.build_loaders import build_loaders
from utils.build_model import build_model
from models.hungary import HungarianMatcher
from utils.logger import logger


if __name__ == '__main__':
    
    device = torch.device(cfgs.device if torch.cuda.is_available() else 'cpu')

    train_loader, valid_loader, test_loader = build_loaders(cfgs)

    hungary_matcher = HungarianMatcher()
    model = build_model(cfgs)
    model = model.float()
    model.to(device)
    print("seed: %d" % cfgs.seed)
    logger.info("seed: %d" % cfgs.seed)
    logger.info("msg: %s" % cfgs.msg)
    print("max_epochs: %d" % cfgs.train.max_epochs)
    logger.info("cfgs.SSG: %s" % cfgs.SSG)
    logger.info("cfgs.use_SSE: %s" % cfgs.use_SSE)
    logger.info("cfgs.decoder.use_ISE: %s" % cfgs.decoder.use_ISE)
    logger.info("cfgs.test.beam_size: %s" % cfgs.test.beam_size)
    # model = train_fn(cfgs, cfgs.model_name, model, hungary_matcher, train_loader, valid_loader, device)
    checkpoints_path_folder = cfgs.train.checkpoints_dir

    if cfgs.data.dataset_name == "MSVD":
        beam_alpha_list = [0.5]
    elif cfgs.data.dataset_name == "MSRVTT":
        beam_alpha_list = [0.8]
    else:
        raise RuntimeError
    for beam_alpha in beam_alpha_list:
        cfgs.args.beam_alpha = beam_alpha
        logger.info("-------------------------------------------")
        logger.info("beam_alpha: %.2f" % beam_alpha)
        logger.info("-------------------------------------------")
        for root, dirs, files in os.walk(checkpoints_path_folder):
            files.sort(reverse=True)
            for file in files:
                file_path = os.path.join(root, file)
                print(file_path)
                cfgs.train.file_path = file_path
                logger.info(file_path)
                model.load_state_dict(torch.load(file_path))
                model.eval()
                test_fn(cfgs, model, test_loader, device)

