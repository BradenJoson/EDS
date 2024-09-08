import torch
import pickle
from collections import OrderedDict, defaultdict

from utils.coco_caption.pycocoevalcap.bleu.bleu import Bleu
from utils.coco_caption.pycocoevalcap.cider.cider import Cider
from utils.coco_caption.pycocoevalcap.meteor.meteor import Meteor
from utils.coco_caption.pycocoevalcap.rouge.rouge import Rouge
from configs.settings import TotalConfigs

import os
import time

# add
from configs.settings import get_settings
import numpy as np
from utils.logger import logger


cfgs = get_settings()
prediction_dict_path = os.path.join(cfgs.test.result_dir, 'prediction_dict.pkl')
reference_dict_path = os.path.join(cfgs.test.result_dir, 'reference_dict.pkl')
vids_list_path = os.path.join(cfgs.test.result_dir, 'vids_list.pkl')


def language_eval(sample_seqs, groundtruth_seqs):
    assert len(sample_seqs) == len(groundtruth_seqs), 'length of sampled seqs is different from that of groundtruth seqs!'

    references, predictions = OrderedDict(), OrderedDict()
    for i in range(len(groundtruth_seqs)):
        references[i] = [groundtruth_seqs[i][j] for j in range(len(groundtruth_seqs[i]))]
    for i in range(len(sample_seqs)):
        predictions[i] = [sample_seqs[i]]

    predictions = {i: predictions[i] for i in range(len(sample_seqs))}
    references = {i: references[i] for i in range(len(groundtruth_seqs))}
    avg_bleu_score, bleu_score = Bleu(4).compute_score(references, predictions)
    print('avg_bleu_score == ', avg_bleu_score)
    avg_cider_score, cider_score = Cider().compute_score(references, predictions)
    print('avg_cider_score == ', avg_cider_score)
    avg_meteor_score, meteor_score = Meteor().compute_score(references, predictions)
    print('avg_meteor_score == ', avg_meteor_score)
    avg_rouge_score, rouge_score = Rouge().compute_score(references, predictions)
    print('avg_rouge_score == ', avg_rouge_score)
    
    logger.info('avg_bleu_score == {}'.format(avg_bleu_score))
    logger.info('avg_cider_score == {}'.format(avg_cider_score))
    logger.info('avg_meteor_score == {}'.format(avg_meteor_score))
    logger.info('avg_rouge_score == {}'.format(avg_rouge_score))

    return {'BLEU': avg_bleu_score, 'CIDEr': avg_cider_score, 'METEOR': avg_meteor_score, 'ROUGE': avg_rouge_score}


def decode_idx(seq, itow, eos_idx):
    ret = ''
    length = seq.shape[0]
    for i in range(length):
        if seq[i] == eos_idx: break
        if i > 0: ret += ' '
        ret += itow[seq[i]]
    return ret


@torch.no_grad()
def eval_fn(model, loader, device, idx2word, save_on_disk, cfgs: TotalConfigs, vid2groundtruth)->dict:
    model.eval()
    if save_on_disk:
        prediction_dict = {}
        reference_dict = {}
    predictions, gts, vids_list = [], [], []
    total_time = 0

    for i, (feature2ds, feature3ds, objects_feats, objects_masks, \
            vp_semantics, caption_semantics, numberic_caps, masks, \
            captions, nouns_dict_list, vids, vocab_ids, vocab_probs, fillmasks) \
            in enumerate(loader):
        feature2ds = feature2ds.to(device)
        feature3ds = feature3ds.to(device)
        objects_feats = objects_feats.to(device)
        objects_masks = objects_masks.to(device)
        vp_semantics = vp_semantics.to(device)
        caption_semantics = caption_semantics.to(device)
        numberic_caps = numberic_caps.to(device)
        masks = masks.to(device)
        if save_on_disk:
            time1 = time.time()
            pred, seq_probabilities = model.sample(cfgs, model, objects_feats, objects_masks, feature2ds, feature3ds, device=device)
            time2 = time.time()
            total_time += (time2-time1)
            print("the beam search time is %.6f" % (time2-time1))
            # logger.info("the beam search time is %.6f" % (time2-time1))
            print("the total time is %.6f" % total_time)
            # logger.info("the total time is %.6f" % total_time)
        else:
            pred, seq_probabilities = model.sample(cfgs, model, objects_feats, objects_masks, feature2ds, feature3ds, device=device)

        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        assert isinstance(pred, np.ndarray)
        all_beam_predictions = []
        if cfgs.test.topk == 1:
            batch_pred = [decode_idx(seq[0], idx2word, cfgs.dict.eos_idx) for seq in pred]
            for one_cap in batch_pred:
                all_beam_predictions.append([one_cap])
        elif cfgs.test.topk > 1:
            batch_pred = [decode_idx(seqs[0], idx2word, cfgs.dict.eos_idx)  for seqs in pred]
            for seqs in pred:
                all_beam_predictions.append([decode_idx(single_seq, idx2word, cfgs.dict.eos_idx) for single_seq in seqs])
        predictions += batch_pred
        # all_predictions += all_batch_pred
        batch_gts = [vid2groundtruth[id] for id in vids] if save_on_disk else [item for item in captions]
        gts += batch_gts

        vids_list += vids

        if save_on_disk:
            assert len(batch_pred) == len(vids), \
                'expect len(batch_pred) == len(vids), ' \
                'but got len(batch_pred) == {} and len(vids) == {}'.format(len(batch_pred), len(vids))
            for vid, pred, gt in zip(vids, all_beam_predictions, batch_gts):
                prediction_dict[vid] = pred
                reference_dict[vid] = gt

    score_states = language_eval(sample_seqs=predictions, groundtruth_seqs=gts)
    if save_on_disk:
        txt_path = os.path.join(cfgs.test.result_dir, '{file_name}_test_CIDEr_{cider_score}.txt'
                        .format(file_name = cfgs.train.file_path.split('/')[-1], cider_score = score_states['CIDEr']))
        with open(txt_path, 'a') as f_txt:
            f_txt.write('avg_bleu_score == {}\n'.format(score_states['BLEU']))
            f_txt.write('avg_cider_score == {}\n'.format(score_states['CIDEr']))
            f_txt.write('avg_meteor_score == {}\n'.format(score_states['METEOR']))
            f_txt.write('avg_rouge_score == {}\n'.format(score_states['ROUGE']))
        if score_states['CIDEr'] > cfgs.test.base_cider:
            with open(prediction_dict_path, 'wb') as f:
                pickle.dump(prediction_dict, f)
            with open(reference_dict_path, 'wb') as f:
                pickle.dump(reference_dict, f)
            with open(vids_list_path, 'wb') as f:
                pickle.dump(vids_list, f)
            cfgs.test.base_cider = score_states['CIDEr']

    return score_states
