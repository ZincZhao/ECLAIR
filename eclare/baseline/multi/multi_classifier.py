# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-27 18:41
import functools
import logging
from typing import Union, List, Callable, Dict

import torch
from elit.utils.util import reorder

from eclare.baseline.rchilli_util import input_from_rchilli
from eclare.metrics.confusion_matrix import ConfusionMatrix
from elit.common.constant import IDX
from elit.utils.time_util import CountdownTimer
from elit.common.dataset import PadSequenceDataLoader, SortingSampler
from torch.utils.data import DataLoader
from elit.layers.transformers.pt_imports import AutoTokenizer, AutoConfig
from eclare.baseline.multi.multi_class_dataset import MultiClassDataset, create_features
from eclare.baseline.multi.multi_model import BertForResumeClassification
from elit.common.torch_component import TorchComponent
from elit.common.vocab import Vocab


class MultiClassifier(TorchComponent):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tokenizer = None

    def build_dataloader(self, data, batch_size, shuffle=False, device=None, logger: logging.Logger = None,
                         **kwargs) -> DataLoader:
        dataset = MultiClassDataset(data, transform=[input_from_rchilli,
                                                     functools.partial(create_features, tokenizer=self._tokenizer),
                                                     self.vocabs], cache=isinstance(data, str))
        if dataset.cache:
            timer = CountdownTimer(len(dataset))
        lens = []
        for idx, sample in enumerate(dataset):
            lens.append(len(sample['lines_input_ids']))
            if dataset.cache:
                # noinspection PyUnboundLocalVariable
                timer.log('Pre-processing and caching dataset [blink][yellow]...[/yellow][/blink]',
                          ratio_percentage=None)
        return PadSequenceDataLoader(dataset, batch_sampler=SortingSampler(lens, batch_size=batch_size),
                                     pad={'attention_masks': 0}, device=device)

    def build_optimizer(self, **kwargs):
        pass

    def build_criterion(self, **kwargs):
        pass

    def build_metric(self, **kwargs):
        return ConfusionMatrix()

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, ratio_width=None, **kwargs):
        pass

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger, **kwargs):
        pass

    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, metric=None, output=False, logger=None,
                            **kwargs):
        self.model.eval()
        timer = CountdownTimer(len(data))
        total_loss = 0
        metric.reset()
        if output:
            predictions = []
            orders = []
            samples = []
        for batch in data:
            outputs = self.feed_batch(batch)
            prediction = self.decode(outputs[0])
            metric(prediction, batch['label_id'])
            if output:
                predictions.extend(prediction.tolist())
                orders.extend(batch[IDX])
                samples.extend(list(zip(batch['sent_a'], batch['sent_b'])))
            total_loss += outputs[1].item()
            timer.log(self.report_metrics(total_loss / (timer.current + 1), metric), ratio_percentage=None,
                      logger=logger)
            del outputs
        if output:
            predictions = reorder(predictions, orders)
            samples = reorder(samples, orders)
            with open(output, 'w') as out:
                for s, p in zip(samples, predictions):
                    out.write('\t'.join(s + (str(p),)))
                    out.write('\n')
        return total_loss / timer.total

    def build_model(self, training=True, **kwargs) -> torch.nn.Module:
        # noinspection PyTypeChecker
        return BertForResumeClassification(AutoConfig.from_pretrained('bert-base-uncased'))

    def predict(self, data: Union[Dict, List[Dict]], batch_size: int = 4, **kwargs):
        """ Predict the CRC level.

        Args:
            data: Sentence pairs.
            batch_size: The number of samples in a batch.
            **kwargs: Not used.

        Returns:
            Similarities between sentences.
        """
        if not data:
            return []
        flat = isinstance(data, dict)
        if flat:
            data = [data]
        dataloader = self.build_dataloader(data,
                                           batch_size=batch_size or self.config.batch_size,
                                           device=self.device)
        orders = []
        predictions = []
        for batch in dataloader:
            output_dict = self.feed_batch(batch)
            prediction = self.decode(output_dict[0], output_probs=True)
            probs = [dict(zip(self.vocabs['label'].idx_to_token, x)) for x in prediction.tolist()]
            predictions.extend([{
                "best_prediction": [max(p.items(), key=lambda t: t[1])],
                "predictions": p
            } for p in probs])
            orders.extend(batch[IDX])
        predictions = reorder(predictions, orders)
        if flat:
            return predictions[0]
        return predictions

    def convert(self, save_dir):
        self.model = BertForResumeClassification.from_pretrained(save_dir)
        self.model.config.output_hidden_states = self.model.config.output_attentions = False
        self.config.transformer = 'bert-base-uncased'
        self.config.batch_size = 4
        self.vocabs['label'] = Vocab(token_to_idx={'NQ': 0, 'CRCI': 1, 'CRCII': 2, 'CRCIII': 3, 'CRCIV': 4},
                                     unk_token='NQ', pad_token='NQ')
        self.vocabs.lock()

    def on_config_ready(self, **kwargs):
        super().on_config_ready(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def feed_batch(self, batch: dict):
        return self.model(types_lines_ids=batch['lines_input_ids'], attention_masks=batch['attention_masks'],
                          types_encoding_ids=batch['type_input_ids'], labels=batch.get('label_id'))

    def decode(self, logits, output_probs=False):
        if output_probs:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return probs
        return logits.argmax(-1, keepdim=True)

    def report_metrics(self, loss, metric):
        return f'loss: {loss:.4f} {metric}'
