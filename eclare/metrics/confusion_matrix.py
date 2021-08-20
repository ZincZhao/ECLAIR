# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-27 21:13
import torch

from elit.metrics.metric import Metric


class ConfusionMatrix(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.total_predictions = torch.zeros(0)
        self.total_gold_labels = torch.zeros(0)

    @property
    def score(self):
        return float((self.total_predictions == self.total_gold_labels).sum()) / self.total_gold_labels.size(0)

    def __call__(self, predictions, gold_labels, mask=None):
        predictions = predictions.reshape(-1)
        gold_labels = gold_labels.reshape(-1)
        predictions = predictions.detach().to(self.total_predictions)
        gold_labels = gold_labels.detach().to(self.total_gold_labels)
        self.total_predictions = torch.cat((self.total_predictions, predictions), 0)
        self.total_gold_labels = torch.cat((self.total_gold_labels, gold_labels), 0)

    def reset(self):
        self.total_predictions = torch.zeros(0)
        self.total_gold_labels = torch.zeros(0)

    def __str__(self) -> str:
        return f'Accuracy: {self.score:.2%}'
