import math
import string
import torch
from torch import nn

N_EMBD = 768         # How many components each token vector have
N_LAYER = 12         # How many blocks of attention + feed forward stacked on each other
N_HEAD = 12          # How many parallel data streams are the input split into during attention
BLOCK_SIZE = 512     # Context window, how far back the model looks to determine its response
DROPOUT = 0.1        # How often a neuron is randomly turned off to combat overfitting
BATCH_SIZE = 128     # How many sentences the model processes at once
LEARNING_RATE = 1e-4 # How much the model updates its weights at each learning step
DEVICE = 'mps'       # Apple's metal performance shaders framework, equivalent to cuda

CHARS = ["<PAD>"] + list(string.printable) # A list of integers, english characters (upper + lower), and symbols. as well as a pad token for empty spaces
VOCAB_SIZE = 50257                         # GPT-2 tokenizer vocab size, originally was len(CHARS) but decided to switch from character-level from scratch tokenizer to tiktoken last minute


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit

    A better activation than ReLU for LLMs, it uses a probability rather than a hard gate. causing a smooth curve and making x=0 differentiable
    """

    def forward(self, x):
        """
        Args:
            x: Number (integer or float) input

        Returns:
            The approximated GELU output (y value)
        """

        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Dropout(nn.Module):
    """
    Dropout refers to the process of setting neurons to 0 (turning them off) randomly, this makes sure the model doesn't overrely on a specific neuron to prevent overfitting
    """

    def __init__(self, p):
        """
        Args:
            p: The probability that a dropout will happen for a neuron
        """

        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Args:
            x: The data tensor flowing through attention/feed forward

        Returns:
            The masked data tensor

        Logic:
            1. A mask of 0s and 1s are created, 0s are determined by the probability self.p
            2. The mask is applied to the input tensor, effectively setting the "unlucky" components to 0
        """

        if self.training:
            mask = (torch.rand_like(x) > self.p).float()

            return (x * mask) / (1 - self.p)
        return x


class LayerNorm(nn.Module):
    """
    Layer Normalization

    Process in which the variance of values are compressed down to avoid astronomically huge/tiny values after series of multiplications
    """

    def __init__(self, ndim, eps=1e-5):
        """
        Args:
            ndim: The model/embedding dimension (channel), we only normalize on the channel so we tell pytorch to create a vector of that dimension
            eps: Epsilon, an infinitesimally small number, similar to infinity in concept
        """

        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim))

    def forward(self, x):
        """
        Args:
            x: The entire data tensor from attention/feed forward

        Returns:
            The data tensor with the channel vector normalized

        Logic:
            1. Calculate the mean, variance, and normalized version of the channel vector
            2. Perform a linear projection on the channel vector (dot product + bias)

        Note:
            We use a custom variance equation because torch.var() divides by n-1 but in layer normalization we divide by n
        """

        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class Tokenizer:
    """
    A dictionary/mapping of every character to a unique integer
    """

    def __init__(self):
        """
        Logic:
            1. Creates 2 dictionaries of numbers counting up from 0 that pairs with each object (character) in CHARS and vice versa
        """

        self.stoi = {char: i for i, char in enumerate(CHARS)}
        self.itos = {i: char for i, char in enumerate(CHARS)}

    def encode(self, text):
        """
        Args:
            text: A string input ready to be tokenized

        Returns:
            A list of integers representing each character in the string

        Logic:
            1. A list comprehension that fills up with the integer at every character's index in the dictionary created in __init__
        """

        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, indices):
        """
        Args:
            indices: A list of integers

        Returns:
            A word or sequence of characters

        Logic:
            1. Finds the corresponding character for each integer and joins then into one string
        """

        return ''.join([self.itos[i] for i in indices])


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self Attention

    Mechanism where the model looks back on previous tokens to understand context in order to properly respond. Multi-Head because many instances of these run in parallel and causal means that each token can't "look" at future tokens to "cheat"
    """

    def __init__(self):
        """
        Logic:
            1. Initialize the attention and weights and biases, as well as the output projection weights and biases
            2. Weights are created using random values with low variance, achieved from multiplying by 0.02
            3. One weight is used for the entirety of query, key, and value for computational efficiency, which is why one of its axis is scaled by 3
            4. Dropout objects are created for attention and residuals
            5. A lower triangular causal mask is created and resized to match the post-attention data tensor rank
        """

        super().__init__()

        self.c_attn = nn.Parameter(torch.randn(N_EMBD, N_EMBD * 3) * 0.02)
        self.c_attn_bias = nn.Parameter(torch.zeros(N_EMBD * 3))

        self.c_proj = nn.Parameter(torch.randn(N_EMBD, N_EMBD) * 0.02)
        self.c_proj_bias = nn.Parameter(torch.zeros(N_EMBD))

        self.attn_dropout = Dropout(DROPOUT)
        self.resid_dropout = Dropout(DROPOUT)

        mask = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        self.register_buffer("bias", mask.view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        """
        Args:
            x: The embedding tensor of shape (batch_size (Batch), block_size (Time), n_embd (Channel))

        Returns:
            a tensor of same shape with context embedded into it

        Logic:
            1. The specific batch, time, and channel values are extracted from the current tensor for better flexibility opposed to using the hardcoded hyperparameters
            2. The input embedding tensor is projected by the attention weight matrix and bias vector: (B, T, C) @ (C, 3C) + (3C) -> (B, T, 3C)
            3. The qkv projection is split into each query, key, and value tensor along the channel vector: (B, T, 3C) -> Q: (B, T, C), K: (B, T, C), V: (B, T, C)
            4. The query, key, and value tensors are then reshaped to accommodate for the parallel attention heads: q, k, v: (B, T, C) -> (B, T, Heads (H), Head Dimension (D)). They are then transposed along the 2nd and 3rd axis because PyTorch performs matrix multiplication on higher rank tensors by effectively looping/ignoring the first few axis and only performing dot products on the last 2. Therefore, to achieve the desired shape of (B, H, T, T) for our similarity tensor (because our similarity matrix [which is inside the tensor] should be a square matrix of every single token of the batch [so that the insides of the similarity matrix grid are the similarity score between each "outside" word]), we would have to make the last 2 axis include T (General rule of matrix multiplication is that the 2 "outer" axis are preserved).
            5. Calculate the attention scores, do this by first transposing the 3rd and 4th vectors of the key tensor: (B, H, T, D) -> (B, H, D, T) (because of that general rule of matrix multiplication).
            6. Perform a matrix multiplication on the transposed key and original query, then divide by the square root of the head dimension (scaling factor) to lower the variance: (B, H, T, D) @ (B, H, D, T) * (head_dim)^-0.5 -> (B, H, T, T)
            7. Mask the similarity tensor using the causal mask, replacing the upper triangle with negative infinities, because later in softmax we raise e to each value in the tensor, and e to the power of negative infinity is 0
            8. Softmax the similarity matrix, turning it into a probability distribution, then apply the attention dropout
            9. Now matrix multiply the probability distribution tensor with the value tensor, to get a distribution of how related each token in the value is to the query: (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
            10. Combine and reassemble the heads, start by undoing the transposition: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
            11. Output projection and dropout: (B, T, C) @ (C, C) -> (B, T, C)
        """

        B, T, C = x.shape

        qkv = x @ self.c_attn + self.c_attn_bias

        head_dim = C // N_HEAD
        q = qkv[:, :, :C].view(B, T, N_HEAD, head_dim).transpose(1, 2)
        k = qkv[:, :, C:2 * C].view(B, T, N_HEAD, head_dim).transpose(1, 2)
        v = qkv[:, :, 2 * C:].view(B, T, N_HEAD, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        att = att - att.max(dim=-1, keepdim=True)[0]
        att = att.exp()
        att = att / att.sum(dim=-1, keepdim=True)

        att = self.attn_dropout(att)

        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = y @ self.c_proj + self.c_proj_bias
        y = self.resid_dropout(y)

        return y


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed Forward Network)

    Here is where the model stores its "information". the 4x projection is a massive tensor of weights that all represent the model's knowledge. Attention will give a word context, but the model won't know how to use the context correctly simply because it doesn't know what the context itself means. The MLP helps the model interpret what the context means and lets it appropriately generate a response
    """

    def __init__(self):
        """
        Logic:
            1. Initialize the up and down projection weights and biases
            2. Create the activation function and dropout object
        """

        super().__init__()
        self.c_fc = nn.Parameter(torch.randn(N_EMBD, 4 * N_EMBD) * 0.02)
        self.c_fc_bias = nn.Parameter(torch.zeros(4 * N_EMBD))

        self.c_proj = nn.Parameter(torch.randn(4 * N_EMBD, N_EMBD) * 0.02)
        self.c_proj_bias = nn.Parameter(torch.zeros(N_EMBD))

        self.act = GELU()
        self.dropout = Dropout(DROPOUT)

    def forward(self, x):
        """
        Args:
            x: the data tensor, shape: (B (batch/batch_size), T (time/block_size), C (channel/n_embd))

        Returns:
            The data tensor of same shape, but instead of just a data tensor of tokens having context, the model now "understands" the context

        Logic:
            1. Project the input data tensor and the up weight and bias: (B, T, C) @ (C, 4C) -> (B, T, 4C)
            2. Activate to introduce nonlinearity (allows model to learn)
            3. Down project the activated tensor: (B, T, 4C) @ (4C, C) -> (B, T, C)
            4. Dropout to prevent overreliance on a specific neuron and to prevent overfitting
        """

        h = self.act(x @ self.c_fc + self.c_fc_bias)
        h = h @ self.c_proj + self.c_proj_bias
        return self.dropout(h)


class Block(nn.Module):
    """
    The Decoder Block

    In the original transformer architecture they included an Encoder and Decoder block, modern LLMs only use decoder blocks because we use autoregressive models, meaning we repeatedly feed the model the data/chat over and over again as it generates step by step. The decoder block is an assembly of layer normalization, attention, and the feed forward
    """

    def __init__(self):
        """
        Logic:
            1. Initialize the 2 layer normalization objects (normalization is where we reduce variance to prevent massive/tiny values after multiplication)
            2. Initialize the attention and mlp objects
        """

        super().__init__()
        self.ln_1 = LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln_2 = LayerNorm(N_EMBD)
        self.mlp = MLP()

    def forward(self, x):
        """
        Args:
            x: The input token embedding tensor

        Returns:
            The context aware/understood embedding tensor

        Logic:
            1. Normalize the embedding tensor (known as pre-norm, it's more preferred than post-norm [normalization after attention/feed forward])
            2. Pass the normalized tensor through attention and add to original embedding (residual connections, helps the model remember the original embedding, but more importantly solves the "Vanishing Gradient Problem". It's where during backpropagation (learning/training) the gradients (partial derivatives) will get smaller and smaller because of how the matrix multiplications in each layer often have low variance (normalization). This causes a tiny gradient/error signal for the weight at the start of the model, causing barely any learning. To solve this we use addition and the original embedding to essentially add 1 to the gradient in each layer, (the derivative of x is 1) to keep it from becoming tiny)
            3. Normalize the post-attention embedding and pass it through the mlp
        """

        # Pre-Norm Architecture
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    Generative Pre-trained Transformer

    This is where our blocks are stacked/assembled, and where embeddings are created
    """

    def __init__(self):
        """
        Logic:
            1. Create the token embedding weights, again multiplying by 0.02 to reduce variance. token embedding is a vector of floating point numbers that correspond to a token
            2. Create the position embedding weights, position embeddings are so the model knows the order of the tokens
            3. Initialize dropout object, layer normalization object, and hidden layers (decoder blocks)
        """

        super().__init__()
        self.wte = nn.Parameter(torch.randn(VOCAB_SIZE, N_EMBD) * 0.02)
        self.wpe = nn.Parameter(torch.randn(BLOCK_SIZE, N_EMBD) * 0.02)

        self.dropout = Dropout(DROPOUT)

        self.h = nn.ModuleList([Block() for _ in range(N_LAYER)])

        self.ln_f = LayerNorm(N_EMBD)

    def forward(self, idx):
        """
        Args:
            idx: The token matrix

        Returns:
            A tensor of scores for each token for generation/response

        Logic:
            1. First make sure the input context length isn't longer than the model expects, if it is then crop out the earlier tokens to fit
            2. Create a vector of integers counting from 0 for all tokens
            3. Get token embeddings with the token matrix and position embeddings with the integer vector just created. tok_emb: (B, T, C) pos_emb: (T, C)
            4. Add the 2 embeddings together, using broadcasting. In linear algebra you can't add 2 tensors of varying rank so pytorch combats this by matching dimensions from the "right" and fills in for missing axis (in our case the B axis is implicitly 1 in our pos_emd)
            5. Pass it through all blocks, then finally normalize again (because our block uses pre-norm)
            6. Matrix multiply the logits by transpose of the token embedding weights to project the channel dimension to the entire vocab size, for probability calculation later: (B, T, C) @ (C VOCAB_SIZE) -> (B, T, VOCAB_SIZE)
        """

        B, T = idx.shape

        if T > BLOCK_SIZE:
            idx = idx[:, -BLOCK_SIZE:]
            T = BLOCK_SIZE

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.wte[idx]
        pos_emb = self.wpe[pos]

        x = self.dropout(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = x @ self.wte.t()

        return logits