import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"
class Global_Graph(nn.Module):
    def __init__(self, hidden_size):
        super(Global_Graph, self).__init__()
        self.reduce = nn.Linear(1000, 7)
        self.l1 = nn.Linear(8, 16)
        self.l2 = nn.Linear(16, 3)


    def forward(self, hidden_states, attention_mask=None, mapping=None):
        print('Hidden States Global')
        print(hidden_states)
        print('Attention Mask Global')
        print(attention_mask)
        print('Mapping Global')
        print(mapping)
        out = torch.relu(self.reduce(hidden_states))
        out = torch.relu(self.l1(out))
        out = self.l2(out)
        return out


class Sub_Graph(nn.Module):
    def __init__(self, hidden_size, depth=3):
        super(Sub_Graph, self).__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size//2)
        self.l2 = nn.Linear(hidden_size // 2, hidden_size)
        self.agg = nn.MaxPool1d(hidden_size, stride=1)
        self.l3 = nn.Linear(16, hidden_size)

    def forward(self, hidden_states, lengths):
        print('Hidden States Sub')
        print(hidden_states[0][1])
        print(hidden_states.shape)
        print('Lengths Sub')
        print(lengths)
        out = torch.relu(self.l1(hidden_states))
        out = torch.relu(self.l2(out))
        #out_agg = self.agg(out)
        #out = torch.cat((out, out_agg))
        return out
