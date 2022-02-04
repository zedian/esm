import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class TransformerEncoder(nn.Module):
    """
    Transformer encoder block (MultiheadAttention -> LayerNorm -> FF1 -> FF2 -> LayerNorm)
    
    
    """
    
    def __init__(
        self,
        embedding_dim,
        ff_intermediate_dim,
        num_heads,
        attention_dropout,
        output_dropout,
        add_bias_kv
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ff_intermediate_dim = ff_intermediate_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.add_bias_kv = add_bias_kv
        
        self.q_dim, self.k_dim, self.v_dim = embedding_dim
        
        self.q = nn.Linear(self.q_dim, embedding_dim)
        self.k = nn.Linear(self.k_dim, embedding_dim)
        self.v = nn.Linear(self.v_dim, embedding_dim)
        
        # Note: torch multiheadattention needs QKV input
        self.multihead_attention = nn.MultiheadAttention(embedding_dim, num_heads, attention_dropout, add_bias_kv = add_bias_kv)
        
        self.attn_layer_norm = nn.LayerNorm(embedding_dim)
        
        self.fc1 = nn.Linear(self.embedding_dim, self.ff_intermediate_dim)
        self.fc2 = nn.Linear(self.ff_intermediate_dim, self.embedding_dim)
                
        self.output_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, 
        x, 
        mask = None
    ):
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        
        attention = self.multihead_attention(query, key, value)
        
        normed_sum = self.attn_layer_norm(attention + x)
        
        x = self.fc1(normed_sum)
        x = self.fc2(x)
        
        normed_output = self.output_layer_norm(x)
        return normed_output
        
class MaskedLM(nn.Module):
    def __init__(
        self,
        embedding_dim,
        embedding_weights
    ):
        super(MaskedLM, self).__init__()
        
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.act = gelu()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.decoder = nn.Linear(embedding_weights.size(1), 
                                 embedding_weights.size(0))
        self.decoder.weight = embedding_weights
        self.bias = nn.Parameter(torch.zeros(embedding_weights.size(0)))
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.layer_norm(x)
        
        x = self.decoder(x) + self.bias
        return x
        
        
        