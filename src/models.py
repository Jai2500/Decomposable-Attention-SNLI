import torch
import torch.nn as nn
import pytorch_lightning as pl

class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size, param_init, intra_sent_atten=False):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.param_init = param_init
        self.intra_sent_atten = intra_sent_atten

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)

        if self.intra_sent_atten:
            self.mlp_f = None
            raise NotImplementedError

        self.input_linear = nn.Linear(self.embedding_size, self.hidden_size, bias=False)


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.param_init)

    def forward(self, sent1, sent2):
        batch_size = sent1.size(0)

        sent1 = self.embedding(sent1)
        sent2 = self.embedding(sent2)

        sent1 = sent1.view(-1, self.embedding_size)
        sent2 = sent2.view(-1, self.embedding_size)

        sent1_linear = self.input_linear(sent1).view(batch_size, -1, self.hidden_size)
        sent2_linear = self.input_linear(sent2).view(batch_size, -1, self.hidden_size)

        return sent1_linear, sent2_linear

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, param_init=None):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        if param_init is not None:
            self._init_params(param_init)

    def _init_params(self, param_init):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, param_init)
                m.bias.data.normal_(0, param_init)

    def forward(self, x):
        return self.seq(x)

class Atten(nn.Module):
    def __init__(self, hidden_size, label_size, param_init):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.param_init = param_init

        self.mlp_f = self._get_mlp_layers(self.hidden_size, self.hidden_size)
        self.mlp_g = self._get_mlp_layers(2 * self.hidden_size, self.hidden_size)
        self.mlp_h = self._get_mlp_layers(2 * self.hidden_size, self.hidden_size)

        self.final_linear = nn.Linear(self.hidden_size, self.label_size)

        self.log_prob = nn.LogSoftmax()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.param_init)
                m.bias.data.normal_(0, self.param_init)
    
    def _get_mlp_layers(self, input_dim, output_dim):
        return nn.Sequential(
            MLP(input_dim, output_dim, self.param_init),
            MLP(output_dim, output_dim, self.param_init)
        )

    def forward(self, sent1_linear, sent2_linear):
        len1 = sent1_linear.size(1)
        len2 = sent2_linear.size(1)

        '''Attend'''
        f1 = self.mlp_f(sent1_linear.view(-1, self.hidden_size))
        f2 = self.mlp_f(sent2_linear.view(-1, self.hidden_size))

        f1 = f1.view(-1, len1, self.hidden_size) # batch_size x len1 x hidden_size
        f2 = f2.view(-1, len2, self.hidden_size) # batch_size x len2 x hidden_size

        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2)) # e_{ij}
        prob1 = torch.nn.functional.softmax(score1.view(-1, len2)).view(-1, len1, len2)
        
        score2 = torch.transpose(score1.contiguous(), 1, 2) # e_{ji}
        score2 = score2.contiguous()
        prob2 = torch.nn.functional.softmax(score2.view(-1, len1)).view(-1, len2, len1)


        sent1_combine = torch.cat([
            sent1_linear, torch.bmm(prob1, sent2_linear)
        ], 2) # batch_size x len1 x (2 * hidden_size)
        sent2_combine = torch.cat([
            sent2_linear, torch.bmm(prob2, sent1_linear)
        ], 2) # batch_size x len2 x (2 * hidden_size)


        '''Sum'''

        g1 = self.mlp_g(sent1_combine.view(-1, 2 * self.hidden_size))
        g1 = g1.view(-1, len1, self.hidden_size)
        g2 = self.mlp_g(sent2_combine.view(-1, 2 * self.hidden_size))
        g2 = g2.view(-1, len2, self.hidden_size)

        sent1_output = torch.sum(g1, 1)
        sent2_output = torch.sum(g2, 1)

        input_combine = torch.cat([sent1_output, sent2_output], 1)

        h = self.mlp_h(input_combine)
        
        h = self.final_linear(h)

        log_prob = self.log_prob(h)

        return log_prob
