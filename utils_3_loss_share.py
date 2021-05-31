# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  
label_map = {'PER':0, 'LOC':1, 'ORG':2, 'GPE':3, 'O':4}


def build_dataset(config):

    def load_dataset(path, entity_pad_size=128, context_pad_size = 128):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                entity_content, context_content, label, maskpos = lin.split('\t')
                entity_content = entity_content.strip()
                context_content = context_content.strip()

                label = label_map[label]
                entity_token = config.tokenizer_share.tokenize(entity_content)
                context_token = config.tokenizer_share.tokenize(context_content)
                entity_token = [CLS] + entity_token
                context_token = [CLS] + context_token
                entity_seq_len = len(entity_token)
                context_seq_len = len(context_token)
                entity_mask = []
                context_mask = []
                entity_token_ids = config.tokenizer_share.convert_tokens_to_ids(entity_token)
                context_token_ids = config.tokenizer_share.convert_tokens_to_ids(context_token)
                #
                # print(context_content)
                # print(context_token)
                if context_seq_len > 128:
                    continue

                maskpos = context_token.index('[MASK]')

                if entity_pad_size:
                    if len(entity_token) < entity_pad_size:
                        entity_mask = [1] * len(entity_token_ids) + [0] * (entity_pad_size - len(entity_token))
                        entity_token_ids += ([0] * (entity_pad_size - len(entity_token)))
                    else:
                        entity_mask = [1] * entity_pad_size
                        entity_token_ids = entity_token_ids[:entity_pad_size]
                        entity_seq_len = entity_pad_size
                if context_pad_size:
                    if len(context_token) < context_pad_size:
                        context_mask = [1] * len(context_token_ids) + [0] * (context_pad_size - len(context_token))
                        context_token_ids += ([0] * (context_pad_size - len(context_token)))
                    else:
                        context_mask = [1] * context_pad_size
                        context_token_ids = context_token_ids[:context_pad_size]
                        context_seq_len = context_pad_size

                contents.append((entity_token_ids, entity_seq_len, entity_mask, context_token_ids, context_seq_len, context_mask, int(label), int(maskpos)))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        entity_token_ids = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        entity_seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        entity_mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        context_token_ids = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        context_seq_len = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        context_mask = torch.LongTensor([_[5] for _ in datas]).to(self.device)

        label = torch.LongTensor([_[6] for _ in datas]).to(self.device)
        maskpos = torch.LongTensor([_[7] for _ in datas]).to(self.device)
        return (entity_token_ids, entity_seq_len, entity_mask), (context_token_ids, context_seq_len, context_mask, maskpos), label

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):

    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
