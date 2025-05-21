import torch
import torch.nn as nn
from REC.utils import InputType
from REC.model.basemodel import BaseModel
from REC.model.layers import TransformerEncoder
import torch.nn.functional as F
import math
import numpy as np
from vllm import LLM, SamplingParams
import time
import time
import pickle
import os

#torch.set_default_dtype(torch.float64)
class SRGNNSEQLLM(BaseModel):
    input_type = InputType.AUGSEQ
    def __init__(self, config, data):
        super(SRGNNSEQLLM, self).__init__()

        self.device = config['device']
        self.item_num = data.item_num

        # set up GNN
        self.item_hidden_size = config['item_embedding_size']  
        self.gnn_step = config['gnn_step'] # number of hidden GNN

        self.item_embedding = nn.Embedding(self.item_num, self.item_hidden_size)
        self.gnn = GNN(self.item_hidden_size, step=self.gnn_step)

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

        # set-up lm model
        self.llm_cache = {}
        self.llm_embed_size = config['llm_embed_dim']
        self.graph_llm = FrozenGraphLLM(config['query_model'], config['encoder_model'], device = self.device, gpu_mem_utl = config['llm_gpu_utlization'])
        self.llm_linear = nn.Linear(self.llm_embed_size, self.seq_hidden_size)

        self.cross_attention = MultiHeadCrossAttention(
            n_heads = self.seq_n_heads,
            hidden_size = self.seq_hidden_size,
            hidden_dropout_prob = self.seq_hidden_dropout_prob,
            attn_dropout_prob = self.seq_attn_dropout_prob,
            layer_norm_eps = self.seq_layer_norm_eps,
        )

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

    def seq_modeling(self, alias_inputs, gnn_output, llm_output, mask):
        llm_output = llm_output.to(alias_inputs.device)
        # transformer: output
        seq_input = []
        llm_input = []
        
        for i in range(len(alias_inputs)):
            seq_inputi = gnn_output[0, alias_inputs[i]]
            seq_input.append(seq_inputi)

            llm_input.append(llm_output[alias_inputs[i]])

        seq_input = torch.stack(seq_input)
        llm_input = self.llm_linear(torch.stack(llm_input))

        # cross-attention seq_input and llm_input
        seq_input = self.cross_attention(seq_input, llm_input, self.seq_get_attention_mask(mask, bidirectional=True))
        print(f'seq_input.shape = {seq_input.shape}')

        # position embedding
        position_ids = torch.arange(mask.size(1), dtype=torch.long, device=mask.device)
        position_ids = position_ids.unsqueeze(0).expand_as(mask)
        position_embedding = self.position_embedding(position_ids)

        seq_input_emb = seq_input + position_embedding
        seq_input_emb = self.seq_layernorm(seq_input_emb)
        seq_input_emb = self.seq_dropout(seq_input_emb)
        extended_attention_mask = self.seq_get_attention_mask(mask, bidirectional=False)

        seq_output_emb = self.trm_encoder(
            seq_input_emb, extended_attention_mask, output_all_encoded_layers=False
        )

        return seq_output_emb[-1]

    def forward(self, input):
        _, global_inputs, _, _, A_b, global_items, mask, tarpos, tarneg = input

        # look up table map id -> embedding vector
        hidden = self.item_embedding(global_items)
        target_pos_embs = self.item_embedding(tarpos)
        target_neg_embs = self.item_embedding(tarneg)

        # Stage 1: apply GNN to get node embeddings (incoporating local interaction signals)
        gnn_output = self.gnn(A_b, hidden)

        # Stage 2: query graph llm model for graph structure details
        batch_key = str(A_b.tolist())

        if batch_key not in self.llm_cache:
            llm_output = self.graph_llm.query(A_b)
            self.llm_cache[batch_key] = 'tmp/llm_output_' + str(time.time()) + '.pkl'
            with open(self.llm_cache[batch_key], 'wb') as f:
                pickle.dump(llm_output, f)
        else:
            with open(self.llm_cache[batch_key], 'rb') as f:
                llm_output = pickle.load(f)

        llm_output.to(global_inputs.device)
        seq_output = self.seq_modeling(global_inputs, gnn_output, llm_output, mask)

        pos_score = (seq_output * target_pos_embs).sum(-1)
        neg_score = (seq_output * target_neg_embs).sum(-1)

        loss = -(torch.log((pos_score - neg_score).sigmoid() + 1e-8) * mask).sum(-1)
        return loss.mean(-1)

    @torch.no_grad()
    def predict(self, input, item_feature):
        _, global_inputs, _, _, A_b, global_items, mask = input

        hidden = item_feature[global_items]
        gnn_output = self.gnn(A_b, hidden)
        llm_output = self.graph_llm.query(A_b)
        
        seq_output = self.seq_modeling(global_inputs, gnn_output, llm_output, mask)

        scores = torch.matmul(seq_output[:, -1], item_feature.t())
        return scores

    @torch.no_grad()
    def compute_item_all(self):
        embed_item = self.item_embedding.weight
        return embed_item

    def seq_get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -1e9)

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

class FrozenGraphLLM:
    def __init__(self, query_model, encoder_model, device, gpu_mem_utl = 0.2, res_max_token = 200, res_temp = 0.8, res_top_p = 0.95):
        world_size = int(os.environ["WORLD_SIZE"])
        print(f'world_size = {world_size}')
        
        self.device = device
        self.sampling_params = SamplingParams(temperature=res_temp, top_p=res_top_p, max_tokens = res_max_token)
        self.query_model = LLM(model=query_model, gpu_memory_utilization = gpu_mem_utl, tensor_parallel_size = world_size)
        self.encoder_model = LLM(model=encoder_model, gpu_memory_utilization = gpu_mem_utl, enforce_eager=True, tensor_parallel_size = world_size)

    def query(self, A_b):
        num_vertice = A_b.shape[1]
        A_in, A_out = A_b[:, :, :num_vertice].squeeze(0), A_b[:, :, num_vertice:].squeeze(0)
        A = ((A_in + A_out) != 0).float()
        
        prompts = []
        # constructing prompt
        for i in range(A.shape[0]):
            prompts.append(f'You are an expert in graph modeling. The row {i} of the graph adjacency matrix is {A[i,:]}. \
                             Describe the relationship of vertex {i} with the remaining vertices. \
                             Include any topological properties if applicable.')

        # perform the inference
        outputs = self.query_model.generate(prompts, self.sampling_params)
        responses = []

        # query the LLM
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            responses.append(generated_text)


        print(f'A[1,:] = {A[1,:]}')
        print(f'Example response for vertex 1: {responses[1]}')

        # convert responses to embeddings
        embeddings = []
        encoded_outputs = self.encoder_model.encode(responses)
        for eo in encoded_outputs:
            embeddings.append(torch.tensor(eo.outputs.embedding))

        return torch.stack(embeddings)

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head Cross-attention layers, a attention score dropout layer is introduced.

    Args:
        item_seq (torch.Tensor): the first sequence of the multi-head cross-attention layer
        embed_seq (torch.Tensor): the second sequence of the multi-head cross-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadCrossAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, item_seq, embed_seq, attention_mask):
        mixed_query_layer = self.query(item_seq)
        mixed_key_layer = self.key(item_seq)
        mixed_value_layer = self.value(embed_seq)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + embed_seq)

        return hidden_states