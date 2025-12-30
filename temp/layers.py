import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dimension, dropout_probability, block_size):
        super().__init__()

        self.qkv_projection = nn.Linear(model_dimension, model_dimension * 3, bias=False)
        self.output_projection = nn.Linear(model_dimension, model_dimension, bias=False)
        self.dropout = nn.Dropout(dropout_probability)

        mask = torch.tril(torch.ones(block_size, block_size).unsqueeze(0).unsqueeze(0))
        mask = mask.masked_fill(mask == 0, -1e9).masked_fill(mask == 1, 0.0)
        self.register_buffer("causal_mask", mask)

    def forward(self, input_tensor, number_heads):
        qkv = self.qkv_projection(input_tensor)
        batch_size, sequence_length, dimension3 = qkv.size()
        head_dimension = dimension3 // (3 * number_heads)
        qkv = qkv.reshape(batch_size, sequence_length, 3, number_heads, head_dimension)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        similarity = torch.matmul(query, key.transpose(-2, -1))
        scaled_similarity = similarity / (head_dimension ** 0.5)
        masked_scaled_similarity = scaled_similarity + self.causal_mask[:, :, :sequence_length, :sequence_length]

        attention_weights = torch.softmax(masked_scaled_similarity, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, value)

        attention_output = attention_output.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
        attention_output = self.output_projection(attention_output)

        return attention_output

class FeedForward(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()

        self.input_layer = nn.Linear(model_dimension, model_dimension * 4)
        self.output_layer = nn.Linear(model_dimension * 4, model_dimension)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, input_tensor):
        pre_activation = self.input_layer(input_tensor)
        post_activation = self.activation(pre_activation)
        output_tensor = self.output_layer(post_activation)
        output_tensor = self.dropout(output_tensor)

        return output_tensor