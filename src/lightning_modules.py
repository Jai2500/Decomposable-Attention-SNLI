from typing import ForwardRef
import torch
import pytorch_lightning as pl
import torchmetrics
from src.data_models import *
from src.models import *

class LitSNLI(pl.LightningDataModule):
    def __init__(self, train_fname, val_fname, test_fname, max_length, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
        self.train_fname = train_fname
        self.val_fname = val_fname
        self.test_fname = test_fname
        self.max_length = max_length

    def setup(self, stage = None):
        self.train_dataset = SNLI(self.train_fname, self.max_length)
        self.val_dataset = SNLI(self.val_fname, self.max_length)
        self.test_dataset = SNLI(self.test_fname, self.max_length)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=1)
    
class LitModel(pl.LightningModule):
    def __init__(self, encoder, atten, max_grad_norm, lr, optim, weight_decay):
        super().__init__()
        self.encoder = encoder
        self.atten = atten
        self.criterion = torch.nn.NLLLoss()
        self.acc = torchmetrics.Accuracy()
        
        self.lr = lr
        self.optim = optim
        self.weight_decay = weight_decay

        self.max_grad_norm = max_grad_norm

    def forward(self, sent1, sent2):
        sent1_linear, sent2_linear = self.encoder(sent1, sent2)
        return self.atten(sent1_linear, sent2_linear)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.optim == 'adam':
            return torch.optim.Adam(params, lr=self.lr)
        elif self.optim == 'adagrad':
            return torch.optim.Adagrad(params, lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        source, target, labels = batch['source'], batch['target'], batch['labels']
        log_prob = self(source, target)
        loss = self.criterion(log_prob, labels)
        self.log("Train Loss", loss.item(), prog_bar=True)

        acc = self.acc(torch.exp(log_prob), labels)
        self.log("Train Accuracy", acc.item(), prog_bar=True)

        return loss

    def on_after_backward(self) -> None:
        grad_norm = 0
        para_norm = 0

        for m in self.encoder.modules():
            if isinstance(m, torch.nn.Linear):
                grad_norm += m.weight.grad.data.norm() ** 2
                para_norm += m.weight.data.norm() ** 2
                if m.bias:
                    grad_norm += m.bias.grad.data.norm() ** 2
                    para_norm += m.bias.data.norm() ** 2


        for m in self.atten.modules:
            if isinstance(m, torch.nn.Linear):
                grad_norm += m.weight.grad.data.norm() ** 2
                para_norm += m.weight.data.norm() ** 2
                if m.bias:
                    grad_norm += m.bias.grad.data.norm() ** 2
                    para_norm += m.bias.data.norm() ** 2

        shrinkage = self.max_grad_norm / grad_norm
        if shrinkage < 1:
            for m in self.encoder.modules():
                if isinstance(m, torch.nn.Linear()):
                    m.weight.grad.data = m.weight.grad.data * shrinkage
            for m in self.atten.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.grad.data = m.weight.grad.data * shrinkage
                    m.bias.grad.data = m.bias.grad.data * shrinkage

    def validation_step(self, batch, batch_idx):
        source, target, labels = batch['source'], batch['target'], batch['labels']
        labels = labels[0]
        log_prob = self(source, target)
        print(labels)
        print(log_prob)
        loss = self.criterion(log_prob, labels)
        self.log("Validation Loss", loss.item(), prog_bar=True)

        acc = self.acc(torch.exp(log_prob), labels)
        self.log("Validation Accuracy", acc.item(), prog_bar=True)

        return loss
