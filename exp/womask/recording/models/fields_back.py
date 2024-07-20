import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from models.embedder import get_embedder
from icecream import ic
from einops import repeat
import math

class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        pos_mlp_hidden_dim=64,
        attn_mlp_hidden_mult=4,
        num_neighbors=None
    ):
        super().__init__()
        self.num_neighbors = num_neighbors

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def forward(self, x, pos, mask = None):
        n = x.shape[1]

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # calculate relative positional embeddings
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None, :] - k[:, None, :, :]

        # expand values
        v = repeat(v, 'b j d -> b i j d', i = n)

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # attention
        attn = sim.softmax(dim = -2)

        # aggregate
        agg = jt.linalg.einsum('b i j d, b i j d -> b i d', attn, v)
        return agg

class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):

        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
         
        resetgate = jt.sigmoid(i_r.add(h_r))
        inputgate = jt.sigmoid(i_i.add(h_i))
        newgate = jt.tanh(i_n.add(resetgate.multiply(h_n)))
        
        hy = newgate.add(inputgate.multiply(hidden - newgate))
        
        return hy

class attention(nn.Module):
    """
    An implementation of GRUCell.

    """
    def __init__(self, hidden_size, bias=True):
        super(attention, self).__init__()
    
        self.a = nn.Parameter(jt.zeros((hidden_size, hidden_size)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        
        e = nn.leaky_relu(jt.matmul(x, self.a), scale=0.2)
        attention =nn.softmax(e, dim=1)
        graph_pooling = attention * x
        
        return graph_pooling

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return jt.sin(30 * input)

def sine_init(m):
    with jt.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with jt.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)