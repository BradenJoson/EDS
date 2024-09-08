import torch
import torch.nn as nn
from models.caption_models.caption_module import CaptionModule
from configs.settings import TotalConfigs
import pickle
from utils import utils
import os
# import time


class HierarchicalModel(CaptionModule):
    def __init__(self, entity_level: nn.Module, predicate_level: nn.Module, sentence_level: nn.Module,
                 decoder: nn.Module, word_embedding_weights, max_caption_len: int, beam_size: int, cfgs: TotalConfigs, 
                 pad_idx=0, temperature=1, eos_idx=0, sos_idx=-1, unk_idx=-1):
        """
        Args:
            entity_level: for encoding objects information.
            predicate_level: for encoding action information.
            sentence_level: for encoding the whole video information.
            decoder:  for generating words.
            word_embedding_weights: pretrained word embedding weight.
            max_caption_len: generated sentences are no longer than max_caption_len.
            pad_idx: corresponding index of '<PAD>'.
        """
        super(HierarchicalModel, self).__init__(beam_size=beam_size)
        self.entity_level = entity_level
        self.predicate_level = predicate_level
        self.sentence_level = sentence_level
        self.decoder = decoder
        self.max_caption_len = max_caption_len
        self.temperature = temperature
        self.eos_idx = eos_idx
        self.sos_idx = sos_idx
        self.unk_idx = unk_idx
        self.use_SSE = cfgs.use_SSE
        hidden_dim = cfgs.decoder.hidden_dim
        semantics_dim = cfgs.encoder.semantics_dim

        self.toHidden =  nn.Linear(hidden_dim + semantics_dim, hidden_dim, bias=False)

        if self.use_SSE:
            self.device = cfgs.device
            train_search_path = os.path.join(cfgs.data.language_dir, "./train_search_{}.pkl".format(cfgs.data.dataset_name))
            with open(train_search_path, 'rb') as f:
                train_search = pickle.load(f)
            self.train_semantics = torch.tensor(train_search["train_semantics"]).to(self.device)
            self.train_captions = train_search["train_captions"]
            self.top_k = cfgs.encoder.top_k
        self.__init_weight()
        
    def __init_weight(self):
        init_range = 0.1
        self.toHidden.weight.data.uniform_(-init_range, init_range)


    def _search_sim_semantices(self, video_semantics): 
        batchsize = video_semantics.shape[0]
        semantics_dim = video_semantics.shape[1]
        cos_sim = utils.cos_sim(video_semantics, self.train_semantics)
        sentences_sim, sentences_idx = torch.topk(cos_sim, k=self.top_k, dim=1)
        mix_sim_semantics = torch.index_select(self.train_semantics, dim=0, index=sentences_idx.flatten()).view(batchsize, self.top_k, semantics_dim)
        video_sim = torch.ones_like(sentences_sim) - sentences_sim 
        mix_sim_semantics = sentences_sim.unsqueeze(2) * mix_sim_semantics + video_sim.unsqueeze(2) * video_semantics.unsqueeze(1).repeat(1, self.top_k, 1)
        mix_sim_semantics = torch.sum(mix_sim_semantics, dim=1)
        return mix_sim_semantics
    

    def forward_encoder(self, objects, objects_mask, feature2ds, feature3ds):
        """

        Args:
            objects: (bsz, max_objects_per_video, object_dim)
            objects_mask: (bsz, max_objects_per_video)
            feature2ds: (bsz, sample_numb, feature2d_dim)
            feature3ds: (bsz, sample_numb, feature3d_dim)

        Returns:
            objects_feats: (bsz, max_objects, hidden_dim)
            action_feats: (bsz, sample_numb, hidden_dim)
            video_feats: (bsz, sample_numb, hidden_dim)

            objects_semantics: (bsz, max_objects, word_dim) (bsz, max_objects, semantics_dim)
            action_semantics: (bsz, semantics_dim)
            video_semantics: (bsz, semantics_dim)
        """
        objects_feats, objects_semantics = self.entity_level(feature2ds, feature3ds, objects, objects_mask)
        action_feats, action_semantics = self.predicate_level(feature3ds, objects_feats, objects_mask, objects_semantics)
        video_feats, video_semantics = self.sentence_level(feature2ds, action_feats, objects_feats, objects_mask, objects_semantics, action_semantics)
        if self.use_SSE:
            # start_time = time.time()
            mix_sim_semantics = self._search_sim_semantices(video_semantics)
            # print("search_time:", time.time()-start_time)

        objects_feats_hidden = torch.cat([objects_feats, torch.mean(objects_semantics, dim=1, keepdim=True).repeat(1, objects_feats.size(1), 1)], dim=-1)
        action_feats_hidden = torch.cat([action_feats, action_semantics.unsqueeze(1).repeat(1, action_feats.size(1), 1)], dim=-1)
        if self.use_SSE:
            video_feats_hidden = mix_sim_semantics.unsqueeze(1).repeat(1, video_feats.shape[1], 1)
            video_feats_hidden = torch.cat([video_feats, video_feats_hidden], dim=-1)
        else:
            video_feats_hidden = torch.cat([video_feats, video_semantics.unsqueeze(1).repeat(1, video_feats.size(1), 1)], dim=-1)
        enc_output = torch.cat([objects_feats_hidden, action_feats_hidden, video_feats_hidden], dim=1)
        enc_output = self.toHidden(enc_output)


        return objects_feats, action_feats, video_feats, objects_semantics, action_semantics, video_semantics, enc_output


    def forward_decoder(self, tgt_seq, decoding_type, output_attentions, **kwargs):
        """

        Args:
            objects_feats: (bsz, max_objects, hidden_dim)
            action_feats: (bsz, sample_numb, hidden_dim)
            video_feats: (bsz, sample_numb, hidden_dim)
            objects_semantics: (bsz, max_objects, word_dim)
            action_semantics: (bsz, semantics_dim)
            video_semantics: (bsz, semantics_dim)

            pre_embedding: (bsz, word_embed_dim)
            pre_state: (hidden_state, cell_state)

        Returns:
            output_prob: (bsz, n_vocab)
            current_state: (hidden_state, cell_state)
        """
        output_prob, current_state = self.decoder(tgt_seq=tgt_seq, decoding_type=decoding_type, output_attentions=output_attentions, **kwargs)
        return output_prob, current_state

    def forward(self, objects_feats, objects_mask, feature2ds, feature3ds, numberic_captions):
        """

        Args:
            numberic_captions: (bsz, max_caption_len)

        Returns:
            ret_seq: (bsz, max_caption_len, n_vocab)
        """
        objects_feats, action_feats, video_feats, objects_semantics, action_semantics, video_semantics, enc_output = self.forward_encoder(objects_feats, objects_mask, feature2ds, feature3ds)

        inputs_for_decoder = {'enc_output': enc_output}
        inputs_for_decoder["category"] = None

        tgt_seq = numberic_captions.clone()
        
        output_word, pre_states = self.forward_decoder(
            tgt_seq=tgt_seq,  
            decoding_type='ARFormer',  
            output_attentions=False,
            **inputs_for_decoder 
            )

        return output_word, objects_semantics, action_semantics, video_semantics

