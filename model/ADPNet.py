import torch
import torch.nn as nn
from model.layers import Embedding, VisualProjection, CQAttention, \
    ConditionedPredictor, SCDM, GlobalAttention, rescale, WeightedPool, \
            TextGuidedCluesMiner, SequentialQueryAttention, PositionalEncoding, VideoTextConcatenate
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np

def build_optimizer_and_scheduler(model, configs):
    no_decay = ['bias', 'layer_norm', 'LayerNorm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                configs.num_train_steps)
    return optimizer, scheduler


class ADPNet(nn.Module):
    def __init__(self, configs, word_vectors):
        super(ADPNet, self).__init__()
        self.configs = configs
        self.word_embedding = Embedding(num_words=configs.word_size, num_chars=configs.char_size, out_dim=configs.dim,
                                       word_dim=configs.word_dim, char_dim=configs.char_dim, word_vectors=word_vectors,
                                       drop_rate=configs.drop_rate)
        self.sentence_embedding = WeightedPool(dim = configs.dim)
        self.video_affine = VisualProjection(visual_dim=configs.video_feature_dim, dim=configs.dim,
                                                drop_rate=configs.drop_rate)
        self.audio_affine = VisualProjection(visual_dim=configs.audio_feature_dim, dim=configs.dim,
                                                drop_rate=configs.drop_rate)
        self.sqan = SequentialQueryAttention(nse=configs.nse_num, qdim=configs.dim)
        self.pe = PositionalEncoding(dim=configs.dim)
        self.scdm_v = SCDM(in_dim = configs.dim)
        self.scdm_a = SCDM(in_dim = configs.dim)
        self.cq_attention_v = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concat_v = VideoTextConcatenate(dim=configs.dim)
        self.cq_attention_a = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concat_a = VideoTextConcatenate(dim=configs.dim)
        self.tgcm = TextGuidedCluesMiner(dims=configs.dim)
        self.global_attn_v = GlobalAttention(idim=configs.dim,
                                             odim=configs.dim,
                                             nheads=configs.num_heads//8,
                                             dp=configs.drop_rate/2)
        self.global_attn_av = GlobalAttention(idim=configs.dim,
                                             odim=configs.dim,
                                             nheads=configs.num_heads//8,
                                             dp=configs.drop_rate/2)
        self.predictor_v = ConditionedPredictor(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
                                              max_pos_len=configs.max_pos_len)
        self.predictor_av = ConditionedPredictor(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
                                              max_pos_len=configs.max_pos_len)
        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()
        self.apply(init_weights)

    def forward(self, word_ids, char_ids, video_features, v_mask, audio_features, a_mask, q_mask):
        video_features = self.video_affine(video_features)
        audio_features = self.audio_affine(audio_features)
        word_embeddings = self.word_embedding(word_ids, char_ids)
        sentence_embeddings = self.sentence_embedding(word_embeddings, q_mask)
        se_feats, se_attw = self.sqan(sentence_embeddings, word_embeddings, q_mask)
        video_features = self.pe(video_features)
        audio_features = self.pe(audio_features)
        video_features = self.scdm_v(video_features, v_mask, se_feats, None)
        audio_features = self.scdm_a(audio_features, a_mask, se_feats, None)
        video_features = self.cq_attention_v(video_features, se_feats, v_mask, None)
        video_features = self.cq_concat_v(video_features, sentence_embeddings)
        audio_features = self.cq_attention_a(audio_features, se_feats, a_mask, None)
        audio_features = self.cq_concat_a(audio_features, sentence_embeddings)
        old_video_features = video_features
        audio_features = rescale(audio_features,video_features,self.configs.gpu_idx)
        a_mask = rescale(a_mask,v_mask,self.configs.gpu_idx)
        video_features, audio_features = self.tgcm(self.configs, se_feats, video_features, audio_features, mask=v_mask)
        features = video_features + audio_features
        attn_features, _ = self.global_attn_av(features, v_mask)
        attn_video_features, _ = self.global_attn_v(old_video_features, v_mask)
        start_logits_av, end_logits_av = self.predictor_av(attn_features, mask=v_mask)
        start_logits_v, end_logits_v = self.predictor_v(attn_video_features, mask=v_mask)        
        return start_logits_av, end_logits_av, start_logits_v, end_logits_v, se_attw
   
    def extract_index(self, start_logits, end_logits):
        return self.predictor_av.extract_index(start_logits=start_logits, end_logits=end_logits)
    
    def compute_dqa_loss(self, se_attw):
        NA = se_attw.shape[1]
        r = 0.3
        sub = torch.bmm(se_attw,se_attw.transpose(1,2)) - r*torch.eye(NA).unsqueeze(0).type_as(se_attw)
        P = torch.norm(sub, p="fro", dim=[1,2], keepdim=True)
        da_loss = (P**2).mean()
        return da_loss

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor_av.compute_cross_entropy_loss(start_logits=start_logits, end_logits=end_logits,
                                                         start_labels=start_labels, end_labels=end_labels)
    
    def compute_weighted_loss(self, start_logits, end_logits,
        start_logits_v, end_logits_v, start_labels, end_labels, grades, rho_th, s_v_th):
        rho, s_av, s_v = grades
        start_loss_av = nn.CrossEntropyLoss(reduction='none')(start_logits, start_labels) # not need softmax
        end_loss_av = nn.CrossEntropyLoss(reduction='none')(end_logits, end_labels)
        start_loss_v = nn.CrossEntropyLoss(reduction='none')(start_logits_v, start_labels)
        end_loss_v = nn.CrossEntropyLoss(reduction='none')(end_logits_v, end_labels)
        mask = torch.ones(rho.shape).type_as(rho)
        batch_size = rho.shape[0]
        for i in range(batch_size):
            if rho[i] < rho_th and s_v[i] > s_v_th:
                mask[i] = 0.0
        return ((start_loss_av + end_loss_av)*mask + (start_loss_v+end_loss_v)).mean()