# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-28 12:42
from transformers import BertPreTrainedModel
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import math
from elit.layers.transformers.pt_imports import BertModel


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


class BertForResumeBinaryMHAConcatClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * config.hidden_size, 2)
        self.type_embeddings = nn.Embedding(6, config.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.mha = MultiHeadAttention(config.hidden_size, head_num=2)
        self.init_weights()

    def forward(
            self,
            types_lines_ids=None,
            attention_masks=None,
            types_encoding_ids=None,
            job_description_ids=None,
            job_description_attention_mask=None,
            labels=None,
    ):
        batch_size = types_lines_ids.size(0)
        n_lines = types_lines_ids.size(1)
        sequence_length = types_lines_ids.size(2)
        types_lines_ids = torch.reshape(types_lines_ids, (batch_size * n_lines, sequence_length)),
        attention_masks = torch.reshape(attention_masks, (batch_size * n_lines, sequence_length)),
        outputs = self.bert(input_ids=types_lines_ids[0], attention_mask=attention_masks[0])
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        job_outputs = self.bert(input_ids=job_description_ids, attention_mask=job_description_attention_mask)
        job_description_output = job_outputs[0]
        pooled_job_description_output = job_outputs[1]
        job_description_output = job_description_output.repeat_interleave(repeats=n_lines, dim=0)
        j_to_r_att = self.mha(job_description_output, sequence_output, sequence_output)
        r_to_j_att = self.mha(sequence_output, job_description_output, job_description_output)
        j_to_r_att = j_to_r_att[:, 0]
        r_to_j_att = r_to_j_att[:, 0]
        j_to_r_att = torch.reshape(j_to_r_att, (batch_size, n_lines, self.hidden_size))
        r_to_j_att = torch.reshape(r_to_j_att, (batch_size, n_lines, self.hidden_size))
        types_encoding_ids = torch.reshape(types_encoding_ids, (batch_size, n_lines))
        types_encodings = self.type_embeddings(types_encoding_ids)
        pooled_output = torch.reshape(pooled_output, (batch_size, n_lines, self.hidden_size))
        all_encoding_output = types_encodings + j_to_r_att + r_to_j_att + pooled_output
        all_encoding_output = self.dropout(all_encoding_output)  # (batch_size, n_lines, self.hidden_size)
        all_encoding_output = torch.sum(all_encoding_output, dim=1)  # (batch_size, self.hidden_size)
        all_encoding_output = torch.cat((all_encoding_output, pooled_job_description_output), dim=-1)
        logits = self.classifier(all_encoding_output)
        outputs = (logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            outputs = outputs + (loss,)
        return outputs
