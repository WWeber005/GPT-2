import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    ## formule utilisée:
    ##
    ## attention(Q,K,V) = softmax(Q@K.T/sqrt(dk))@V 
    ##
    
    score = (query @ rearrange(key, 'b h t d -> b h d t')) / (key.shape[3] ** 0.5)

    ## apppliquer un causal mask pour éviter de regarder le future afin de prédire le prochain mot !!
    seq_len = score.size(-1)
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=score.device))
    causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
    score = score.masked_fill(causal_mask == 0, -10000.0)

    ## attention_mask sert a ignorer les mot qui ont été ajouter pour completer la longueur de la phrase étudiée ( PADDING )
    #print(attention_mask)
    #print(f"shape de la matrice d attention {attention_mask.shape}")
    score = score + attention_mask
    ## socre dimension est actuellement: [batch, head, token_actuel, token_regardé]
    score = torch.softmax(score, dim=-1)
    ## dropout après l attention ##
    score = self.dropout(score)
    score = score @ value ## dim=-1 permet de prendre toute les valeurs du tableau score de la dernière dimension
    score = rearrange(score, 'b h t d -> b t h d')
    score = rearrange(score, 'b t h d -> b t (h d)')
    return score
    


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
