import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import (build_model, FeedForwardNetwork, MultiHeadAttention,
                       Parameter, build_norm_layer)

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value

def rescale(src, tgt, gpu):
    tgt_len, src_len = tgt.shape[1], src.shape[1]
    src_rescale = torch.zeros(tgt.shape,device='cuda:{}'.format(gpu))
    for i in range(tgt_len):
        src_rescale[:,i] = src[:,math.floor(i*src_len/tgt_len)]
    return src_rescale

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

def apply_on_sequence(layer, inp):
    inp = to_contiguous(inp)
    inp_size = list(inp.size())
    output = layer(inp.view(-1, inp_size[-1]))
    output = output.view(*inp_size[:-1], -1)
    return output

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term2 = torch.pow(torch.tensor(10000.0),torch.arange(0, dim, 2).float()/dim)
        div_term1 = torch.pow(torch.tensor(10000.0),torch.arange(1, dim, 2).float()/dim)
        pe[:, 0::2] = torch.sin(position * div_term2)
        pe[:, 1::2] = torch.cos(position * div_term1)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Attention(nn.Module):
    def __init__(self, kdim=128, cdim=128, att_hdim=64, drop_p=0.0):
        super(Attention, self).__init__()

        # layers
        self.key2att = nn.Linear(kdim, att_hdim)
        self.feat2att = nn.Linear(cdim, att_hdim)
        self.to_alpha = nn.Linear(att_hdim, 1)
        self.drop = nn.Dropout(drop_p)

    def forward(self, key, feats, feat_masks=None, return_weight=True):
        # check inputs
        assert len(key.size()) == 2, "{} != 2".format(len(key.size()))
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert feat_masks is None or len(feat_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)

        # compute attention weights
        logits = self.compute_att_logits(key, feats, feat_masks) # [B,A]
        weight = self.drop(F.softmax(logits, dim=1))             # [B,A]

        # compute weighted sum: bmm working on (B,1,A) * (B,A,D) -> (B,1,D)
        att_feats = torch.bmm(weight.unsqueeze(1), feats).squeeze(1) # B * D
        if return_weight:
            return att_feats, weight
        return att_feats

    def compute_att_logits(self, key, feats, feat_masks=None):
        # check inputs
        assert len(key.size()) == 2
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert feat_masks is None or len(feat_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)
        A = feats.size(1)

        # embedding key and feature vectors
        att_f = apply_on_sequence(self.feat2att, feats)
        att_k = self.key2att(key)
        att_k = att_k.unsqueeze(1).expand_as(att_f)

        # compute attention weights
        dot = torch.tanh(att_f + att_k)                             # B * A * att_hdim
        alpha = apply_on_sequence(self.to_alpha, dot)     # B * A * 1
        alpha = alpha.view(-1, A)                                   # B * A
        if feat_masks is not None:
            alpha = alpha.masked_fill(feat_masks.float().eq(0), -1e9)

        return alpha

 
class SequentialQueryAttention(nn.Module):
    def __init__(self, nse, qdim):
        super(SequentialQueryAttention, self).__init__()
        self.nse = nse
        self.qdim = qdim
        self.global_emb_fn = nn.ModuleList(
                [nn.Linear(self.qdim, self.qdim) for i in range(self.nse)])
        self.guide_emb_fn = nn.Sequential(*[
            nn.Linear(2*self.qdim, self.qdim),
            nn.ReLU()
        ])
        self.att_fn = Attention(kdim=self.qdim, cdim=self.qdim, 
                                att_hdim=self.qdim // 2, drop_p=0.0)

    def forward(self, q_feats, w_feats, w_mask=None):
        B = w_feats.size(0)
        prev_se = w_feats.new_zeros(B, self.qdim)
        se_feats, se_attw = [], []
        
        for n in range(self.nse):
            q_n = self.global_emb_fn[n](q_feats)
            g_n = self.guide_emb_fn(torch.cat([q_n, prev_se], dim=1))
            att_f, att_w = self.att_fn(g_n, w_feats, w_mask)

            prev_se = att_f
            se_feats.append(att_f)
            se_attw.append(att_w)

        return torch.stack(se_feats, dim=1), torch.stack(se_attw, dim=1)


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        return x.transpose(1, 2)

class SCDM(nn.Module):
    def __init__(self, in_dim):
        super(SCDM, self).__init__()
        self.cal_w = nn.Sequential(
            nn.Linear(in_dim, 1, bias=True),
            nn.Tanh(),)
        self.cal_b = nn.Sequential(
            nn.Linear(in_dim, 1, bias=True),
            nn.Tanh(),)
        self.Ws = nn.Parameter(torch.zeros((in_dim, in_dim)))
        self.W = nn.Parameter(torch.zeros((in_dim, in_dim)))
        self.w  = nn.Parameter(torch.zeros((in_dim, 1)))
        self.b = nn.Parameter(torch.zeros((1, in_dim)))
    
    def forward(self,
                features, masks,
                textual_features, q_masks
                ):
        t_len = textual_features.shape[1]
        b_s = textual_features.shape[0]
        f_len = features.shape[1]
        w_s, b_s = [], []
        for k in range(f_len):
            rho = torch.matmul(
                torch.tanh(
                    torch.matmul(textual_features,self.Ws)
                    + torch.matmul(features[:,k,:],self.W).unsqueeze(1)
                    + self.b.unsqueeze(0)),
                self.w
            ).squeeze()
            if q_masks != None:
                rho = mask_logits(inputs=rho, mask=q_masks)
            rho = torch.softmax(rho, dim=-1).unsqueeze(-1)
            txt_h = torch.sum(rho * textual_features, dim=1)
            w = self.cal_w(txt_h)
            b = self.cal_b(txt_h)
            w_s.append(w)
            b_s.append(b)
        w_s = torch.cat(w_s, dim=-1).unsqueeze(-1)
        b_s = torch.cat(b_s, dim=-1).unsqueeze(-1)
        features = w_s * features + b_s        
        return features

class TextGuidedCluesMinerUnit(nn.Module):
    def __init__(self,
                 dims,
                 heads=8,
                 ratio=4,
                 p=0.1,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(TextGuidedCluesMinerUnit, self).__init__()

        self.dims = dims
        self.heads = heads
        self.ratio = ratio
        self.p = p

        self.att1 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att2 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att3 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att4 = MultiHeadAttention(dims, heads=heads, p=p)

        self.ffn1 = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)
        self.ffn2 = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        self.norm2 = build_norm_layer(norm_cfg, dims=dims)
        self.norm3 = build_norm_layer(norm_cfg, dims=dims)
        self.norm4 = build_norm_layer(norm_cfg, dims=dims)
        self.norm5 = build_norm_layer(norm_cfg, dims=dims)
        self.norm6 = build_norm_layer(norm_cfg, dims=dims)

    def forward(self, a, b, t, pe=None, mask=None):
        da = self.norm1(a)
        db = self.norm2(b)
        dt = self.norm3(t)

        ka = da if pe is None else da + pe
        kb = db if pe is None else db + pe
        
        at, *aw = self.att1(dt, ka, da, mask=mask)
        bt, *bw = self.att2(dt, kb, db, mask=mask)

        t = t + at + bt
        dt = self.norm4(t)

        qa = da if pe is None else da + pe
        qb = db if pe is None else db + pe

        af, *afw = self.att3(qa, dt)
        bf, *bfw = self.att4(qb, dt)
        a = a + af
        b = b + bf
            
        da = self.norm5(a)
        db = self.norm6(b)

        a = a + self.ffn1(da)
        b = b + self.ffn2(db)

        return a, b, t

class TextGuidedCluesMiner(nn.Module):
    def __init__(self, dims, num_tokens=4, num_layers=1, **kwargs):
        super(TextGuidedCluesMiner, self).__init__()
        
        self.dims = dims
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        
        self.token = Parameter(num_tokens, dims)
        self.encoder = nn.ModuleList([
            TextGuidedCluesMinerUnit(dims, **kwargs)
            for _ in range(num_layers)
        ])
        
    def forward(self, configs, q, a, b ,**kwargs):
        for enc in self.encoder:
            a, b, _ = enc(a, b, q, pe=None, **kwargs)
        return a, b
    
class GlobalAttention(nn.Module):
    def __init__(self, idim, odim, nheads, dp):
        super(GlobalAttention, self).__init__()
        self.idim = idim
        self.odim = odim
        self.nheads = nheads
        
        self.use_bias = True
        self.use_local_mask = False
        
        self.c_lin = nn.Linear(self.idim, self.odim*2, bias=self.use_bias)
        self.v_lin = nn.Linear(self.idim, self.odim, bias=self.use_bias)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(dp)

    def forward(self, m_feats, mask):
        mask = mask.float()
        B, nseg = mask.size()

        m_k = self.v_lin(self.drop(m_feats))
        m_trans = self.c_lin(self.drop(m_feats))
        m_q, m_v = torch.split(m_trans, m_trans.size(2) // 2, dim=2)

        new_mq = m_q
        new_mk = m_k

        w_list = []
        mk_set = torch.split(new_mk, new_mk.size(2) // self.nheads, dim=2)
        mq_set = torch.split(new_mq, new_mq.size(2) // self.nheads, dim=2)
        mv_set = torch.split(m_v, m_v.size(2) // self.nheads, dim=2)
        for i in range(self.nheads):
            
            mk_slice, mq_slice, mv_slice = mk_set[i], mq_set[i], mv_set[i]

            m2m = mk_slice @ mq_slice.transpose(1,2) / ((self.odim // self.nheads) ** 0.5)
            
            m2m = m2m.masked_fill(mask.unsqueeze(1).eq(0), -1e9)
            m2m_w = F.softmax(m2m, dim=2)
            w_list.append(m2m_w)

            r = m2m_w @ mv_slice if (i==0) else torch.cat((r, m2m_w @ mv_slice), dim=2)
        
        updated_m = self.drop(m_feats + r)
        return updated_m, torch.stack(w_list, dim=1)

class WordEmbedding(nn.Module):
    def __init__(self, num_words, word_dim, drop_rate, word_vectors=None):
        super(WordEmbedding, self).__init__()
        self.is_pretrained = False if word_vectors is None else True
        if self.is_pretrained:
            self.pad_vec = nn.Parameter(torch.zeros(size=(1, word_dim), dtype=torch.float32), requires_grad=False)
            unk_vec = torch.empty(size=(1, word_dim), requires_grad=True, dtype=torch.float32)
            nn.init.xavier_uniform_(unk_vec)
            self.unk_vec = nn.Parameter(unk_vec, requires_grad=True)
            self.glove_vec = nn.Parameter(torch.tensor(word_vectors, dtype=torch.float32), requires_grad=False)
        else:
            self.word_emb = nn.Embedding(num_words, word_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, word_ids):
        if self.is_pretrained:
            word_emb = F.embedding(word_ids, torch.cat([self.pad_vec, self.unk_vec, self.glove_vec], dim=0),
                                   padding_idx=0)
        else:
            word_emb = self.word_emb(word_ids)
        return self.dropout(word_emb)


class CharacterEmbedding(nn.Module):
    def __init__(self, num_chars, char_dim, drop_rate):
        super(CharacterEmbedding, self).__init__()
        self.char_emb = nn.Embedding(num_chars, char_dim, padding_idx=0)
        kernels, channels = [1, 2, 3, 4], [10, 20, 30, 40]
        self.char_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=char_dim, out_channels=channel, kernel_size=(1, kernel), stride=(1, 1), padding=0,
                          bias=True),
                nn.ReLU()
            ) for kernel, channel in zip(kernels, channels)
        ])
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, char_ids):
        char_emb = self.char_emb(char_ids)
        char_emb = self.dropout(char_emb)
        char_emb = char_emb.permute(0, 3, 1, 2)
        char_outputs = []
        for conv_layer in self.char_convs:
            output = conv_layer(char_emb)
            output, _ = torch.max(output, dim=3, keepdim=False)
            char_outputs.append(output)
        char_output = torch.cat(char_outputs, dim=1)
        return char_output.permute(0, 2, 1)


class Embedding(nn.Module):
    def __init__(self, num_words, num_chars, word_dim, char_dim, drop_rate, out_dim, word_vectors=None):
        super(Embedding, self).__init__()
        self.word_emb = WordEmbedding(num_words, word_dim, drop_rate, word_vectors=word_vectors)
        self.char_emb = CharacterEmbedding(num_chars, char_dim, drop_rate)
        self.linear = Conv1D(in_dim=word_dim + 100, out_dim=out_dim, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, word_ids, char_ids):
        word_emb = self.word_emb(word_ids)
        char_emb = self.char_emb(char_ids)
        emb = torch.cat([word_emb, char_emb], dim=2)
        emb = self.linear(emb)
        return emb


class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        bsz, seq_length = inputs.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings


class VisualProjection(nn.Module):
    def __init__(self, visual_dim, dim, drop_rate=0.0):
        super(VisualProjection, self).__init__()
        self.drop = nn.Dropout(p=drop_rate)
        self.linear = Conv1D(in_dim=visual_dim, out_dim=dim, kernel_size=1, stride=1, bias=True, padding=0)

    def forward(self, visual_features):
        visual_features = self.drop(visual_features)
        output = self.linear(visual_features)
        return output


class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_rate, num_layers=4):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise_separable_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=dim,
                          padding=kernel_size // 2, bias=False),
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(),
            ) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        output = x
        for idx, conv_layer in enumerate(self.depthwise_separable_conv):
            residual = output
            output = self.layer_norms[idx](output)
            output = output.transpose(1, 2)
            output = conv_layer(output)
            output = self.dropout(output)
            output = output.transpose(1, 2) + residual
        return output


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super(MultiHeadAttentionBlock, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)
        output = self.dropout(output)
        query = self.transpose_for_scores(self.query(output))
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(attention_probs, value)
        value = self.combine_last_two_dim(value.permute(0, 2, 1, 3))
        output = self.dropout(value)
        residual = output + x
        output = self.layer_norm2(residual)
        output = self.dropout(output)
        output = self.out_layer(output)
        output = self.dropout(output) + residual
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, kernel_size=7, num_layers=4, drop_rate=0.0):
        super(FeatureEncoder, self).__init__()
        self.pos_embedding = PositionalEmbedding(num_embeddings=max_pos_len, embedding_dim=dim)
        self.conv_block = DepthwiseSeparableConvBlock(dim=dim, kernel_size=kernel_size, drop_rate=drop_rate,
                                                      num_layers=num_layers)
        self.attention_block = MultiHeadAttentionBlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)

    def forward(self, x, mask=None):
        features = x + self.pos_embedding(x)
        features = self.conv_block(features)
        features = self.attention_block(features, mask=mask)
        return features


class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)
        score_ = nn.Softmax(dim=2)(score if q_mask == None else mask_logits(score, q_mask.unsqueeze(1)))
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))
        score_t = score_t.transpose(1, 2)
        c2q = torch.matmul(score_, query)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2
        return res


class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x


class VideoTextConcatenate(nn.Module):
    def __init__(self, dim):
        super(VideoTextConcatenate, self).__init__()
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, sentence_embeddings):
        _, c_seq_len, _ = context.shape
        sentence_embeddings_repeat = sentence_embeddings.unsqueeze(1).repeat(1, c_seq_len, 1)
        output = torch.cat([context, sentence_embeddings_repeat], dim=2)
        output = self.conv1d(output)
        return output


class ConditionedPredictor(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, drop_rate=0.0):
        super(ConditionedPredictor, self).__init__()        
        self.encoder = FeatureEncoder(dim=dim, num_heads=num_heads, kernel_size=7, num_layers=4,
                                          max_pos_len=max_pos_len, drop_rate=drop_rate)
        self.start_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.end_layer_norm = nn.LayerNorm(dim, eps=1e-6)

        self.start_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.end_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, mask):
        start_features = self.encoder(x, mask)
        end_features = self.encoder(start_features, mask)
        start_features = self.start_layer_norm(start_features)
        end_features = self.end_layer_norm(end_features)
        
        start_features = self.start_block(torch.cat([start_features, x], dim=2))
        end_features = self.end_block(torch.cat([end_features, x], dim=2))
        start_logits = mask_logits(start_features.squeeze(2), mask=mask)
        end_logits = mask_logits(end_features.squeeze(2), mask=mask)
        return start_logits, end_logits

    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)
        _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)
        _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)
        return start_index, end_index

    @staticmethod
    def compute_cross_entropy_loss(start_logits, end_logits, start_labels, end_labels):
        start_loss = nn.CrossEntropyLoss(reduction='mean')(start_logits, start_labels)
        end_loss = nn.CrossEntropyLoss(reduction='mean')(end_logits, end_labels)
        return start_loss + end_loss
