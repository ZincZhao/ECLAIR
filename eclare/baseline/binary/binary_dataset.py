# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-28 11:31
from elit.utils.io_util import load_json

from elit.common.dataset import TransformDataset


class BinaryClassDataset(TransformDataset):
    def load_file(self, filepath: str):
        for sample in load_json(filepath):
            yield sample


def create_features(sample: dict, tokenizer, max_line_length=128,
                    doc_stride=0, max_line_number=100,
                    max_job_length=512) -> dict:
    if 'is_matched' in sample:
        sample['label'] = sample['is_matched']
    type_map = {'Profile': 0, 'Skills': 1, 'Work Experience': 2, 'Education': 3, 'Other': 4, 'Activities': 5}
    job_desc = sample["applied_job_description"]
    job_inputs = tokenizer.encode_plus(job_desc, None, add_special_tokens=True, max_length=max_job_length,
                                       truncation=True)
    job_input_ids = job_inputs["input_ids"]
    job_input_tokens = [tokenizer._convert_id_to_token(tid) for tid in job_input_ids]
    attention_mask = [1] * len(job_input_ids)
    padding_length = max_job_length - len(job_input_ids)
    job_input_ids = job_input_ids + ([tokenizer.pad_token_id] * padding_length)
    job_attention_masks = attention_mask + ([0] * padding_length)

    lines_input_ids = []
    type_input_ids = []
    attention_masks = []
    input_lines_tokens = []
    sections = sample.get('resume_data', sample.get('sections', None))
    for section in sections:
        type_s = section['type']
        content = section['content'].replace("\n", " ").replace("\t", " ").replace("\\u", " ").lower().strip()
        type_n = type_map[type_s]
        tokens = tokenizer.tokenize(content)
        if len(tokens) <= max_line_length - 2:
            inputs = tokenizer.encode_plus(content, None, add_special_tokens=True,
                                           max_length=max_line_length, truncation=True)
            input_ids = inputs["input_ids"]
            input_tokens = [tokenizer._convert_id_to_token(tid) for tid in input_ids]
            attention_mask = [1] * len(input_ids)
            padding_length = max_line_length - len(input_ids)
            input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            attention_masks.append(attention_mask)
            lines_input_ids.append(input_ids)
            type_input_ids.append(type_n)
            input_lines_tokens.append(input_tokens)
        else:
            doc_left_index = 0
            doc_right_index = max_line_length - 2
            while doc_left_index < len(tokens) - doc_stride:
                if doc_right_index >= len(tokens):
                    doc_right_index = len(tokens)
                new_line_tokens = tokens[doc_left_index:doc_right_index]
                inputs = tokenizer.encode_plus(" ".join(new_line_tokens), None, add_special_tokens=True,
                                               max_length=max_line_length)
                input_ids = inputs["input_ids"]
                input_tokens = [tokenizer._convert_id_to_token(tid) for tid in input_ids]
                attention_mask = [1] * len(input_ids)
                padding_length = max_line_length - len(input_ids)
                input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                attention_masks.append(attention_mask)
                lines_input_ids.append(input_ids)
                type_input_ids.append(type_n)
                input_lines_tokens.append(input_tokens)
                doc_left_index = doc_right_index - doc_stride
                doc_right_index = doc_left_index + max_line_length - 2
    if len(lines_input_ids) > max_line_number:
        lines_input_ids = lines_input_ids[:max_line_number]
        attention_masks = attention_masks[:max_line_number]
        type_input_ids = type_input_ids[:max_line_number]

    sample['lines_input_ids'] = lines_input_ids
    sample['type_input_ids'] = type_input_ids
    sample['attention_masks'] = attention_masks
    sample['input_lines_tokens'] = input_lines_tokens
    sample['job_input_tokens'] = job_input_tokens
    sample['job_input_ids'] = job_input_ids
    sample['job_attention_masks'] = job_attention_masks
    return sample
