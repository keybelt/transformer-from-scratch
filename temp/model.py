import configuration
import layers
import torch
from torch import nn

from configuration import vocabulary_size


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = layers.MultiHeadAttention(configuration.model_dimension, configuration.dropout_probability, configuration.block_size)
        self.feed_forward = layers.FeedForward(configuration.model_dimension, configuration.dropout_probability)

        self.layer_normalization1 = nn.LayerNorm(configuration.model_dimension)
        self.layer_normalization2 = nn.LayerNorm(configuration.model_dimension)

    def forward(self, input_tensor):
        input_normalized1 = self.layer_normalization1(input_tensor)
        residual_connection1 = input_tensor + self.attention(input_normalized1, configuration.number_heads)

        input_normalized2 = self.layer_normalization2(residual_connection1)
        residual_connection2 = residual_connection1 + self.feed_forward.forward(input_normalized2)

        return residual_connection2

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(configuration.vocabulary_size, configuration.model_dimension)
        self.position_embedding = nn.Embedding(configuration.block_size, configuration.model_dimension)

        self.decoder_stack = nn.ModuleList([DecoderBlock() for _ in range(configuration.number_layers)])
        self.final_layer_norm = nn.LayerNorm(configuration.model_dimension)
        self.lm_head = nn.Linear(configuration.model_dimension, vocabulary_size)

    def forward(self, input_tensor):
        token_embeddings = self.token_embedding(input_tensor)
        position_ids = torch.arange(input_tensor.size(1), device=input_tensor.device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)

        embedding = token_embeddings + position_embeddings

        for decoder_block in self.decoder_stack:
            embedding = decoder_block(embedding)

        embedding = self.final_layer_norm(embedding)
        logits = self.lm_head(embedding)

        return logits