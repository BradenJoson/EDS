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
    
    logger.info("seed: %d" % cfgs.seed)
    logger.info("msg: %s" % cfgs.msg)
    logger.info("cfgs.SSG: %s" % cfgs.SSG)
    logger.info("cfgs.use_SSE: %s" % cfgs.use_SSE)
    logger.info("cfgs.decoder.use_ISE: %s" % cfgs.decoder.use_ISE)
    logger.info("cfgs.test.beam_size: %s" % cfgs.test.beam_size)
    
    #----------------------------------------------------------------------------------------------------#
    # For calculating flops and params
    # batch_size = 1  
    # objects_dim = (batch_size, 120, 2048) 
    # object_masks_dim = (batch_size, 120)
    # feature2ds_dim = (batch_size, 15, 1536)
    # feature3ds_dim = (batch_size, 15, 1024)
    # numberic_caps_dim = (batch_size, 22)  

    # objects = torch.randn(objects_dim).to(device)
    # object_masks = torch.randn(object_masks_dim).to(device)
    # feature2ds = torch.randn(feature2ds_dim).to(device)
    # feature3ds = torch.randn(feature3ds_dim).to(device)
    # numberic_caps = torch.randint(0, 20, (batch_size, 22)).to(device)

    # flops, params = profile(model, inputs=(objects, object_masks, feature2ds, feature3ds, numberic_caps))

    # flops, params = clever_format([flops, params], '%.3f')

    # print(f"FLOPs: {flops}, Params: {params}")
    # logger.info(f"FLOPs: {flops}, Params: {params}")
    #----------------------------------------------------------------------------------------------------#

    model = train_fn(cfgs, cfgs.model_name, model, hungary_matcher, train_loader, valid_loader, device)
    model.load_state_dict(torch.load(cfgs.train.save_checkpoints_path))
    model.eval()
    test_fn(cfgs, model, test_loader, device)

    checkpoints_path_folder = cfgs.train.checkpoints_dir
    
    # beam_alpha_list = [0.5, 0.8, 1.0]
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
            BLEU = 0
            CIDEr = 0
            METEOR = 0
            ROUGE = 0
            for file in files:
                file_path = os.path.join(root, file)
                print(file_path)
                cfgs.train.file_path = file_path
                logger.info(file_path)
                model.load_state_dict(torch.load(file_path))
                model.eval()
                score_states = test_fn(cfgs, model, test_loader, device)
                if score_states['CIDEr'] > CIDEr:
                    BLEU = score_states['BLEU']
                    CIDEr = score_states['CIDEr']
                    METEOR = score_states['METEOR']
                    ROUGE = score_states['ROUGE']
            logger.info('avg_bleu_score == {}\n'.format(BLEU))
            logger.info('avg_cider_score == {}\n'.format(CIDEr))
            logger.info('avg_meteor_score == {}\n'.format(METEOR))
            logger.info('avg_rouge_score == {}\n'.format(ROUGE))
