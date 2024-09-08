import torch
import torch.nn as nn
import numpy as np
import random
import os
from configs import Constants
import json
import shutil
from utils.logger import logger
from torch import Tensor

import pickle
from configs.settings import get_settings
import cv2
from eval import language_eval


cfgs = get_settings()
prediction_dict_path = os.path.join(cfgs.test.result_dir, 'prediction_dict.pkl')
reference_dict_path = os.path.join(cfgs.test.result_dir, 'reference_dict.pkl')
vids_list_path = os.path.join(cfgs.test.result_dir, 'vids_list.pkl')
prediction_dict_HMN_path = os.path.join(cfgs.test.result_dir, 'prediction_HMN_dict.pkl')
topk_dict_path = os.path.join(cfgs.test.result_dir, 'topk_dict.pkl')
sim_dict_path = os.path.join(cfgs.test.result_dir, 'sim_dict.pkl')
MSVD_dataset_path = 'E:/dataset/MSVD/video'
MSRVTT_dataset_path = 'E:/dataset/MSRVTT/video'


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    
    # if a.device != b.device:
    #     b = b.to(a.device)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def to_sentence(hyp, vocab, break_words=[Constants.EOS, Constants.PAD], skip_words=[]):
    sent = []
    for word_id in hyp:
        if word_id in skip_words:
            continue
        if word_id in break_words:
            break
        word = vocab[word_id]
        sent.append(word)
    return ' '.join(sent)


def get_dict_mapping(opt, teacher_opt):
    if teacher_opt is None:
        return {}
    if teacher_opt['vocab_size'] == opt['vocab_size']:
        return {}

    info = json.load(open(opt["info_json"]))
    vocab = info['ix_to_word']

    teacher_info = json.load(open(teacher_opt["info_json"]))
    teacher_vocab = teacher_info['ix_to_word']
    teacher_w2ix = teacher_info['word_to_ix']
    if vocab == teacher_vocab:
        return {}

    dict_mapping = {}
    for k, v in vocab.items():
        dict_mapping[int(k)] = int(teacher_w2ix[v])
    return dict_mapping


def remove_repeat_n_grame(sent, n):
    length = len(sent)
    rec = {}
    result_sent = []
    for i in range(length-n+1):
        key = ' '.join(sent[i:i+n])
        if key in rec.keys():
            dis = i - rec[key] - n
            if dis in [0,1]:
                result_sent += sent[:i-dis]
                if i+n <length:
                    result_sent += sent[i+n:]
                return result_sent, False
        else:
            rec[key] = i
    return sent, True


def duplicate(sent):
    sent = sent.split(' ')
    res = {}
    for i in range(4, 0, -1):
        jud = False
        while not jud:
            sent, jud = remove_repeat_n_grame(sent, i)
            if not jud:
                res[i] = res.get(i, 0) + 1
            else:
                break
    res_str = []
    for i in range(1, 5):
        res_str.append('%d-gram: %d' % (i, res.get(i, 0)))
    return ' '.join(sent), '\t'.join(res_str)


def cal_gt_n_gram(data, vocab, splits, n=1):
    gram_count = {}
    gt_sents = {}
    for i in splits['train']:
        k = 'video%d'% int(i)
        caps = data[k]
        for tmp in caps:
            cap = [vocab[wid] for wid in tmp[1:-1]]
            gt_sents[' '.join(cap)] = gt_sents.get(' '.join(cap), 0) + 1
            for j in range(len(cap)-n+1):
                key = ' '.join(cap[j:j+n])
                gram_count[key] = gram_count.get(key, 0) + 1
    return gram_count, gt_sents


def cal_n_gram(data, n=1):
    gram_count = {}
    sents = {}
    ave_length, count = 0, 0
    for k in data.keys():
        for i in range(len(data[k])):
            sents[data[k][i]['caption']] = sents.get(data[k][i]['caption'], 0) + 1
            cap = data[k][i]['caption'].split(' ')
            ave_length += len(cap)
            count += 1
            for j in range(len(cap)-n+1):
                key = ' '.join(cap[j:j+n])
                gram_count[key] = gram_count.get(key, 0) + 1
    return gram_count, sents, ave_length/count, count


def analyze_length_novel_unique(gt_data, data, vocab, splits, n=1, calculate_novel=True):
    novel_count = 0
    hy_res, hy_sents, ave_length, hy_count = cal_n_gram(data, n)
    if calculate_novel:
        gt_res, gt_sents = cal_gt_n_gram(gt_data, vocab, splits, n)
        for k1 in hy_sents.keys():
            if k1 not in gt_sents.keys():
                novel_count += 1

    novel = novel_count / hy_count
    unique = len(hy_sents.keys()) / hy_count
    vocabulary_usage = len(hy_res.keys())

    gram4, _, _, _ = cal_n_gram(data, n=4)
    return ave_length, novel, unique, vocabulary_usage, hy_res, len(gram4)


def get_words_with_specified_tags(word_to_ix, seq, index_set, demand=['NOUN', 'VERB'], ignore_words=['is', 'are', '<mask>']):
    import nltk
    assert isinstance(index_set, set)
    res = nltk.pos_tag(seq.split(' '))
    for w, t in res:
        if Constants.pos_tag_mapping[t] in demand and w not in ignore_words:
            index_set.add(word_to_ix[w])


def load_satisfied_weights(model, checkpoint_path, str_mapping={}, skip_keys=[], strict=False):
    model_dict = model.state_dict()
    checkpoint_dict = torch.load(checkpoint_path)['state_dict']

    new_state_dict = model_dict
    str_mapping_keys = list(str_mapping.keys())
    str_mapping_values = list(str_mapping.values())

    def check(now_key, all_str):
        for i, item in enumerate(all_str):
            if item in now_key:
                return i
        return -1

    success = 0
    for k in model_dict.keys():
        if k in skip_keys:
            continue
        index = check(now_key=k, all_str=str_mapping_keys)
        if index != -1:
            src, trg = str_mapping_keys[index], str_mapping_values[index]
            key = k.replace(src, trg)
        else:
            key = k

        if key in checkpoint_dict:
            new_state_dict[k] = checkpoint_dict[key]
            success += 1
        else:
            assert not strict, 'key {}/{} can not be found in the checkpoint'.format(k, key)
            new_state_dict[k] = model_dict[k]
    print('Successfully loading {}/{} parameters'.format(success, len(new_state_dict)))
    logger.info('Successfully loading {}/{} parameters'.format(success, len(new_state_dict)))
    
    model.load_state_dict(new_state_dict)
    return model


def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar', best_model_name='best.pth.tar'):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    save_path = os.path.join(filepath, filename) 
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(filepath, best_model_name)
        shutil.copyfile(save_path, best_path)


def enlarge(info, beam_size):
    bsz, *rest_shape = info.shape
    if len(rest_shape) == 2:
        info = info.unsqueeze(1).repeat(1, beam_size, 1, 1)
    elif len(rest_shape) == 1:
        info = info.unsqueeze(1).repeat(1, beam_size, 1)
    else:
        info = info.unsqueeze(1).repeat(1, beam_size)
    return info.view(bsz * beam_size, *rest_shape)


def auto_enlarge(info, beam_size):
    if isinstance(info, list):
        if isinstance(info[0], tuple):
            return [
                tuple([enlarge(_, beam_size) for _ in item])
                for item in info
            ]
        else:
            return [enlarge(item, beam_size) for item in info]
    else:
        if isinstance(info, tuple):
            return tuple([enlarge(item, beam_size) for item in info])
        else:
            return enlarge(info, beam_size)


def visualize_video():
    with open(prediction_dict_path, 'rb') as f:
        prediction_dict = dict(pickle.load(f))
    with open(reference_dict_path, 'rb') as f:
        reference_dict = dict(pickle.load(f))
    with open(topk_dict_path, 'rb') as f:
        topk_dict = dict(pickle.load(f))
    with open(vids_list_path, 'rb') as f:
        vids_list = list(pickle.load(f))
    with open(sim_dict_path, 'rb') as f:
        sim_dict = dict(pickle.load(f))
    with open(prediction_dict_HMN_path, 'rb') as f:
        prediction_HMN_dict = dict(pickle.load(f))
    index = 20
    is_output_word = False
    vids_list_length = len(vids_list)
    if cfgs.data.dataset_name == 'MSRVTT':
        dataset_path = MSRVTT_dataset_path
        music_format = 'mp4'
    elif cfgs.data.dataset_name == 'MSVD':
        dataset_path = MSVD_dataset_path
        music_format = 'avi'
    while True:
        print(os.path.join(dataset_path, '{}.{}'.format(vids_list[index], music_format)))
        cap = cv2.VideoCapture(os.path.join(dataset_path, '{}.{}'.format(vids_list[index], music_format)))
        if not is_output_word:
            print('prediction:')
            for i, sequence in enumerate(prediction_dict[vids_list[index]]):
                print("top%d: %s" % (i+1, sequence))
            print("HMN: ", prediction_HMN_dict[vids_list[index]])
            print("topk_similar_caption:")
            for i in range(len(topk_dict[vids_list[index]])):
                print("top%d: %s %.4f" % (i+1, topk_dict[vids_list[index]][i], sim_dict[vids_list[index]][i]))
            # for i, sequence, sim in enumerate(topk_dict[vids_list[index]], sim_dict[vids_list[index]]):
            #     print("top%d: %s %.4f" % (i+1, sequence, sim))
            print('reference:')
            print(reference_dict[vids_list[index]])
            # prediction_dict[vids_list[index]] is topk beam, select the top1 caption
            language_eval([prediction_dict[vids_list[index]][0]], [reference_dict[vids_list[index]]])
            print('=' * 30)
            is_output_word = True
        while True:
            # Read the current frame
            ret, frame = cap.read()

            # If the frame was read successfully, display it
            if ret == True:
                cv2.imshow('{}.avi'.format(vids_list[index]), frame)

                # Wait for 10 milliseconds, and exit if 'q' is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif cv2.waitKey(10) & 0xFF == ord('s'):
                    # If 's' is pressed, stop the current video and play the next one
                    cap.release()
                    cv2.destroyWindow('{}.avi'.format(vids_list[index]))
                    # index += 1
                    index = random.randint(0, vids_list_length)
                    is_output_word = False
                    break
            else:
                # If the frame was not read successfully, reset the video file to the beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Release the video capture object
        cap.release()