import torch
import torch.nn as nn
import time
from tqdm import tqdm
from .Translator import Translator
import numpy as np



class CaptionModule(nn.Module):
    """
    CaptionModule and its child classes are complementary.
    """
    def __init__(self, beam_size):
        super(CaptionModule, self).__init__()
        self.beam_size = beam_size


    def sample(
            self, cfgs, model,
            objects_feats, objects_mask, feature2ds, feature3ds, 
            device=None, teacher_model=None, dict_mapping={}, category=None
            ):
        if teacher_model is not None:
            teacher_model.eval()
        opt=vars(cfgs.args)
        translator = Translator(opt=opt, model=model, device=device, teacher_model=teacher_model, dict_mapping=dict_mapping)

        all_time = 0
        bsz = cfgs.bsz
        max_caption_len = cfgs.test.max_caption_len
        
        with torch.no_grad():
            objects_feats, action_feats, caption_feats, \
            objects_semantics, action_semantics, caption_semantics, encoder_outputs = self.forward_encoder(objects_feats, objects_mask, feature2ds, feature3ds)


            all_hyp, all_scores = translator.translate_batch_ARFormer(encoder_outputs, category)

            if isinstance(all_hyp, list):
                total_len = len(all_hyp[0])
                for beams in all_hyp:
                    i = 0
                    for item in beams:
                        item.extend([opt['eos_idx']] * (max_caption_len - len(item)))
                        i += 1
                    for step in range(i, total_len):
                        beams.append([opt['eos_idx'] for j in range(max_caption_len)])
                    
                all_hyp = np.array(all_hyp)
        return all_hyp, all_scores


