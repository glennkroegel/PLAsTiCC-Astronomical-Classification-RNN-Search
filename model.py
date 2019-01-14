import pandas as pd
import numpy as np
import pickle
from config import NUM_OUTPUTS

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class std_dense(nn.Module):
    '''A standard dense layer for repeated use.'''
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.1, inplace=True)

    def forward(self, x):
        return self.bn(self.act(self.drop(self.lin(x))))

class RecurrentPassband(nn.Module):
    '''Fully recurrent branch for each passband.
       This module is instantiated 6 times - 1 for each passband below in MergeRNNs'''
    def __init__(self, passband=0):
        super().__init__()
        self.passband = passband
        self.dim_out = 32
        self.nl = 1
        self.rnn = nn.GRU(3, self.dim_out, num_layers=self.nl, bidirectional=False, dropout=0.1, batch_first=True)
        self.pool = nn.AdaptiveMaxPool1d(200)

    def forward(self, seqs):
        x = seqs[:,self.passband]
        bs = len(x)
        dim, sl = x[0].size()

        lens = [a.size(1) for a in x]
        indices = np.argsort(lens)[::-1].tolist()
        rev_ind = [indices.index(i) for i in range(len(indices))]
        x = [x[i] for i in indices]
        x = pad_sequence([a.transpose(0,1) for a in x], batch_first=True)
        input_lengths = [lens[i] for i in indices]
        packed = pack_padded_sequence(x, input_lengths, batch_first=True)
        output, hidden = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[rev_ind, :].contiguous()
        hidden = hidden.transpose(0,1)[rev_ind, :, :].contiguous()
        return output, hidden

class PassbandDecoder(nn.Module):
    '''Takes the hidden state of each RecurrentPassband module and
    reproduces the input signal.
    '''
    pass

class MergeRNNs(nn.Module):
    '''Merges all the RNN passband pipelines into one layer.
       Separate modules to combine the inputs and return object encodings.'''
    def __init__(self):
        super().__init__()
        self.dim_in = RecurrentPassband().dim_out*6
        self.dim_out = 128
        self.nl = 1

        self.pipelines = nn.ModuleList([RecurrentPassband(passband=pb) for pb in range(6)])
        self.rnn1 = nn.GRU(self.dim_in, self.dim_out, num_layers=self.nl, dropout=0.1, batch_first=True)

        # This is the module that will return the encoding to compare objects
        # in a post processing step - see find_unknown_objects.ipynb
        # encoding can be interchanged into something more expressive.
        self.encoding = std_dense(self.dim_in, self.dim_out)

    def forward(self, x):
        bs = x.shape[0]
        outs = torch.cat([module(x)[1] for module in self.pipelines], dim=-1) # Myabe need to pass output
        bs = outs.size(0)
        outs = outs.view(bs, -1) 
        encoding = self.encoding(outs)
        return encoding

class StandardModel(nn.Module):
    '''Output layer of standard model. Either softmax output for classification
       or the encoding from the merging layer can be output. Toggled by get_state flag.'''
    def __init__(self, num_outputs=NUM_OUTPUTS):
        super().__init__()
        self.inputs = MergeRNNs()
        self.input_size = self.inputs.dim_out
        self.num_outputs = num_outputs
        self.l_out = nn.Linear(self.input_size, num_outputs)

    def forward(self, x, get_state=False):
        encoding = self.inputs(x)
        if get_state:
            return encoding
        else:
            output = self.l_out(encoding)
            return output