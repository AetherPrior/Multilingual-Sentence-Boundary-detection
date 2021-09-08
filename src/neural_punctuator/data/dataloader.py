import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

class BertMetrics:
    def __init__(self, prefix, config):
        self.config = config

        with open(config.data.data_path + prefix + "_data.pkl", 'rb') as f:
            texts, targets, space_count, p_count, q_count, comma_count = pickle.load(f)
            self.space_count = space_count
            self.p_count = p_count
            self.q_count = q_count
            self.comma_count = comma_count
    
    def get_metrics(self):
        return self.space_count, self.p_count, self.q_count, self.comma_count

class BertDataset(Dataset):
    def __init__(self, prefix, config, is_train=False):

        self.config = config
        self.is_train = is_train

        with open(config.data.data_path + prefix + "_data.pkl", 'rb') as f:
            texts, targets, _,_,_,_ = pickle.load(f)
            self.encoded_texts = [word for t in texts for word in t]
            self.targets = [t for ts in targets for t in ts]

    def __getitem__(self, idx):

        shift = np.random.randint(self.config.trainer.seq_shift) - self.config.trainer.seq_shift // 2\
            if self.is_train else 0

        start_idx = idx * self.config.model.seq_len + shift
        start_idx = max(0, start_idx)
        end_idx = start_idx + self.config.model.seq_len
        return torch.LongTensor(self.encoded_texts[start_idx: end_idx]),\
               torch.LongTensor(self.targets[start_idx: end_idx])

    def __len__(self):
        return len(self.encoded_texts)//self.config.model.seq_len - 1


def collate(batch):
    texts, targets = zip(*batch)
    return torch.stack(texts), torch.stack(targets)
    # return pad_sequence(batch, batch_first=True, padding_value=PAD_ID)


def get_datasets_metrics(config):
    space_count, p_count, q_count, comma_count = BertMetrics("train",config).get_metrics()
    train_dataset = BertDataset("train", config, is_train=True)
    valid_dataset = BertDataset("valid", config)
    return train_dataset, valid_dataset, space_count, p_count, q_count, comma_count


def get_data_loaders(train_dataset, valid_dataset, config):
    train_loader = DataLoader(train_dataset, batch_size=config.trainer.batch_size, num_workers=0, collate_fn=collate, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.trainer.batch_size, collate_fn=collate)
    return train_loader, valid_loader
