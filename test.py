import torch
import pickle

from configs.settings import TotalConfigs
from eval import eval_fn
from utils.logger import logger


def test_fn(cfgs: TotalConfigs, model, loader, device):
    print('##############n_vocab is {}##############'.format(cfgs.decoder.n_vocab))
    logger.info('##############n_vocab is {}##############'.format(cfgs.decoder.n_vocab))
    with open(cfgs.data.idx2word_path, 'rb') as f:
        idx2word = pickle.load(f)
    with open(cfgs.data.vid2groundtruth_path, 'rb') as f:
        vid2groundtruth = pickle.load(f)
    score_states = eval_fn(model=model, loader=loader, device=device, 
            idx2word=idx2word, save_on_disk=True, cfgs=cfgs, 
            vid2groundtruth=vid2groundtruth)
    print('===================Testing is finished====================')
    logger.info('===================Testing is finished====================')
    return score_states
