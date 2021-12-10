import torch
import numpy as np
import h5py

class SNLI(torch.utils.data.Dataset):
    def __init__(self, fname, max_length):
        super().__init__()

        if max_length < 0:
            max_length = 9999

        self.max_length = max_length

        f = h5py.File(fname, 'r')
        self.source = torch.from_numpy(np.asarray(f['source'])) - 1
        self.target = torch.from_numpy(np.asarray(f['target'])) - 1
        self.label = torch.from_numpy(np.asarray(f['label'])) - 1
        self.label_size = torch.from_numpy(np.asarray(f['label_size']))
        self.source_l = torch.from_numpy(np.asarray(f['source_l']))
        self.target_l = torch.from_numpy(np.asarray(f['target_l'])) # max target length of each batch?
        self.batch_idx = torch.from_numpy(np.array(f['batch_idx'])) - 1
        self.batch_l = torch.from_numpy(np.array(f['batch_l']))

    def __len__(self):
        return self.batch_l.size(0)

    def __getitem__(self, idx):
        # NOTE: This __getitem__ returns the batch directly instead of individual elements
        # NOTE: So use with batch size = 1 in the dataloader

        source_len = min(self.source_l[idx], self.max_length)
        target_len = min(self.target_l[idx], self.max_length)
        #source_len = self.source_l[idx]
        #target_len = self.target_l[idx]

        batch = {
            'source': self.source[self.batch_idx[idx]:self.batch_idx[idx] + self.batch_l[idx]][:, :source_len],
            'target': self.target[self.batch_idx[idx]:self.batch_idx[idx] + self.batch_l[idx]][:, :target_len],
            'labels': self.label[self.batch_idx[idx]:self.batch_idx[idx] + self.batch_l[idx]]
        }
        #print(batch['source'])
        #print(len(batch['source']))

        return batch

class w2v(object):
    def __init__(self, fname) -> None:
        super().__init__()
        f = h5py.File(fname)
        self.word_vecs = torch.from_numpy(np.array(f['word_vecs']))
