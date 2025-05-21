import torch
import torch.nn as nn
from REC.utils import InputType
from REC.model.basemodel import BaseModel
from REC.model.layers import TransformerEncoder
import torch.nn.functional as F
import math
import numpy as np

#torch.set_default_dtype(torch.float64)
class SRGNNSEQ(BaseModel):
    input_type = InputType.AUGSEQ
    def __init__(self, config, data):
        super(SRGNNSEQ, self).__init__()

        self.device = config['device']
        self.item_num = data.item_num

        # set up GNN
        self.item_hidden_size = config['item_embedding_size']  
        self.gnn_step = config['gnn_step'] # number of hidden GNN

        
        self.item_embedding = nn.Embedding(self.item_num, self.item_hidden_size)
        self.gnn = GNN(self.item_hidden_size, step=self.gnn_step)

        # Method 1: set up a simple attention sequence module
        self.linear_one = nn.Linear(self.item_hidden_size, self.item_hidden_size, bias=True)
        self.linear_two = nn.Linear(self.item_hidden_size, self.item_hidden_size, bias=True)
        self.linear_three = nn.Linear(self.item_hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.item_hidden_size * 2, self.item_hidden_size, bias=True)

        # Method 2: set up transfomer sequence
        # load parameters info
        self.seq_n_layers = config['seq_n_layers']
        self.seq_n_heads = config['seq_n_heads']
        self.seq_hidden_size = config['seq_embedding_size']  # same as embedding_size
        self.seq_inner_size = config['seq_inner_size']  # the dimensionality in feed-forward layer  
        self.seq_inner_size *= self.seq_hidden_size
        self.seq_hidden_dropout_prob = config['seq_hidden_dropout_prob']
        self.seq_attn_dropout_prob = config['seq_attn_dropout_prob']
        self.seq_hidden_act = config['seq_hidden_act']
        self.seq_layer_norm_eps = config['seq_layer_norm_eps']
        self.seq_initializer_range = config['seq_initializer_range']
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH'] # problem !!!
        # define layers and loss
        self.seq_item_embedding = nn.Embedding(self.item_num, self.seq_hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.seq_hidden_size)
        
        self.trm_encoder = TransformerEncoder(
            n_layers=self.seq_n_layers,
            n_heads=self.seq_n_heads,
            hidden_size=self.seq_hidden_size,
            inner_size=self.seq_inner_size,
            hidden_dropout_prob=self.seq_hidden_dropout_prob,
            attn_dropout_prob=self.seq_attn_dropout_prob,
            hidden_act=self.seq_hidden_act,
            layer_norm_eps=self.seq_layer_norm_eps
        )

        self.seq_layernorm = nn.LayerNorm(self.seq_hidden_size, eps=self.seq_layer_norm_eps)
        self.seq_dropout = nn.Dropout(self.seq_hidden_dropout_prob)

        # class weight
        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)
        self._gnn_reset_parameters()
        self.apply(self._seq_init_weights)

    def _seq_init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.seq_initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def _gnn_reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.item_hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def seq_modeling(self, alias_inputs, A, gnn_output, mask, items):
        # transformer: output
        seq_input = []
        for i in range(len(alias_inputs)):
            #first_token = gnn_output[i,0].unsqueeze(0) # first token is CLS
            first_token = torch.zeros(1, gnn_output[i].shape[1], device = gnn_output.device)
            seq_inputi = torch.vstack([first_token, gnn_output[i][alias_inputs[i][1:]]])
            
            #indexi = alias_inputs[i].clone()
            #indexi[0] = 0 # first token is [CLS] token
            #seq_inputi = gnn_output[i][indexi]
            
            seq_input.append(seq_inputi)
 
        seq_input = torch.stack(seq_input)
        
        # position embedding
        position_ids = torch.arange(mask.size(1), dtype=torch.long, device=mask.device)
        position_ids = position_ids.unsqueeze(0).expand_as(mask)
        position_embedding = self.position_embedding(position_ids)

        seq_input_emb = seq_input + position_embedding
        seq_input_emb = self.seq_layernorm(seq_input_emb)
        seq_input_emb = self.seq_dropout(seq_input_emb)
        extended_attention_mask = self.seq_get_attention_mask(items,bidirectional=False)        
        
        seq_output_emb = self.trm_encoder(seq_input_emb, extended_attention_mask, output_all_encoded_layers=False)

        return seq_output_emb[-1]

    def forward(self, input):        
        alias_inputs, A, items, mask, targets = input

        # look up table map id -> embedding vector
        hidden = self.item_embedding(items)

        # Stage 1: apply GNN to get node embeddings (incoporating local interaction signals)
        gnn_output = self.gnn(A, hidden)
        #print(f'gnn_ouput.shape = {gnn_output.shape}')

        # Stage 2:Transformer Sequence
        seq_output = self.seq_modeling(alias_inputs, A, gnn_output, mask, items)
        target_output =self.item_embedding(targets)  #[batch,2, dim]

        # compute loss
        score = (seq_output[:,-1:,:] * target_output).sum(-1)
        output = score.view(-1,2)
        batch_loss = -torch.mean(1e-8+torch.log(torch.sigmoid(torch.matmul(output, self.weight))))        
        return batch_loss
      
   
    @torch.no_grad()
    def predict(self, input, item_feature):
        alias_inputs, A, items, mask = input
        hidden = item_feature[items]
        
        gnn_output = self.gnn(A, hidden)
        seq_output = self.seq_modeling(alias_inputs, A, gnn_output, mask, items)

        scores = torch.matmul(seq_output[:,-1:,:],item_feature.t())
        return scores

    @torch.no_grad()    
    def compute_item_all(self):
        embed_item = self.item_embedding.weight
        return embed_item
    
    def seq_get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""        
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -1e9)
        
        return extended_attention_mask
    

class GNN(nn.Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden