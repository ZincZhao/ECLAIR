# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-27 20:18
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel

from elit.layers.transformers.pt_imports import BertModel


class BertForResumeClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 5)
        self.type_embeddings = nn.Embedding(6, config.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

    def forward(
            self,
            types_lines_ids=None,
            attention_masks=None,
            types_encoding_ids=None,
            labels=None,
    ):
        batch_size = types_lines_ids.size(0)
        n_lines = types_lines_ids.size(1)
        sequence_length = types_lines_ids.size(2)
        types_lines_ids = torch.reshape(types_lines_ids, (batch_size * n_lines, sequence_length)),
        attention_masks = torch.reshape(attention_masks, (batch_size * n_lines, sequence_length)),
        outputs = self.bert(input_ids=types_lines_ids[0], attention_mask=attention_masks[0])
        pooled_output = outputs[1]
        assert pooled_output.size(-1) == self.hidden_size
        pooled_output = torch.reshape(pooled_output, (batch_size, n_lines, self.hidden_size))
        types_encoding_ids = torch.reshape(types_encoding_ids, (batch_size, n_lines))
        types_encodings = self.type_embeddings(types_encoding_ids)
        all_encoding_output = types_encodings + pooled_output
        all_encoding_output = self.dropout(all_encoding_output)  # (batch_size, n_lines, self.hidden_size)
        all_encoding_output = torch.sum(all_encoding_output, dim=1)
        logits = self.classifier(all_encoding_output)
        outputs = (logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 5), labels.view(-1))
            outputs = outputs + (loss,)
        return outputs
