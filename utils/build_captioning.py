import torch
import torch.nn as nn
import random
import numpy as np
import os
from configs.settings import get_settings
import pickle
import h5py


cfgs = get_settings()
mode = 'train'
sample_numb = cfgs.sample_numb
max_caption_len = cfgs.test.max_caption_len

# language part
vid2language_path = cfgs.data.vid2language_path
vid2fillmask_path = cfgs.data.vid2fillmask_path

# dataset split part
videos_split_path = cfgs.data.videos_split_path_tpl.format(mode)

with open(videos_split_path, 'rb') as f:
    video_ids = pickle.load(f)


with open(vid2language_path, 'rb') as f:
    vid2language = pickle.load(f)

# build training datasets caption
train_captions = []
train_semantics = []
save_path = "train_search_MSVD.pkl"
for vid in video_ids:
    for item in vid2language[vid]:
        if item[0] not in train_captions:
            train_captions.append(item[0])
            train_semantics.append(item[3])
train_semantics = np.array(train_semantics)
train_search = {"train_captions": train_captions, "train_semantics": train_semantics}
with open('train_search_MSVD.pkl', 'wb') as f:
    pickle.dump(train_search, f)