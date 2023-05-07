''' Define the Transformer model '''

import torch
import torch.nn as nn
import numpy as np
from FeatureEncodeAndDecode import FPN,FDN
from transformer.Layers import EncoderLayer, DecoderLayer
from torch.nn import functional as F


__author__ = "Cheng XinLong, Oxalate-c"

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner,dropout=0.1, n_position=200,scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq,return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq.long())
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output,return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq.long())
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200):

        super().__init__()



        self.d_model = d_model


        self.fpn=FPN()
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers,pad_idx=1, n_head=n_head, d_k=d_k, d_v=d_v,
           dropout=dropout)

        self.fdn = FDN()

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)



        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,input_data,device):
        input_data=self.fpn(input_data).squeeze(-1)
        src_seq=input_data[:,:-1].to(device).to(torch.float32)
        trg_seq=input_data[:,1:].to(device).to(torch.float32)
        trg_mask = get_subsequent_mask(trg_seq)
        enc_output = self.encoder(src_seq)
        dec_output = self.decoder(trg_seq, trg_mask, enc_output)
        trajectory_logit = self.trg_word_prj(dec_output)
        trajectory_logit =self.fdn(trajectory_logit)

        return trajectory_logit

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1_dim, hidden2_dim, use_extra_input=False): # layer size: list of size, input->hidddn->output
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden1_dim)
        self.layer1_ = nn.Linear(input_dim, hidden1_dim) # use for input2
        self.layer2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.layer3 = nn.Linear(hidden2_dim, hidden1_dim)
        self.layer4 = nn.Linear(hidden1_dim, output_dim)
        self.use_extra_input = use_extra_input

    def forward(self, input_data, device):
        input1 = input_data[:,:-1,:].to(device).to(torch.float32).transpose(1,2)
        output = self.layer1(input1)
        if self.use_extra_input:
            input1_ = input_data[:,1:,:].to(device).to(torch.float32).transpose(1,2)
            output += self.layer1_(input1_)
        output = F.relu(output)
        output = F.relu(self.layer2(output))
        output = F.relu(self.layer3(output))
        output = self.layer4(output).transpose(1,2)
        return output





