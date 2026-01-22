import torch
import tiktoken
import time
import math
import os
import numpy as np
from model import GPT, DEVICE, BLOCK_SIZE, LEARNING_RATE

PHYSICAL_BATCH_SIZE = 16                        # The actual amount of sequences the GPU processes at once, because of memory limitations
GRAD_ACCUM_STEPS = 8                            # How many sets of the physical batch size generated gradients are accumulated before weight updates
MAX_ITERS = 35000                               # Total number of training steps
WARMUP_ITERS = 200                              # How many steps the model warms up on, because in the start the gradient is massive from huge errors (model just started learning) so we make the learning rate start from 0 to calm down the step size
MIN_LR = 1e-5                                   # The minimum learning rate the model should use, the learning rate will slowly decrease after warmup to this number
CHECKPOINT_INTERVAL = 2400                      # How many steps before a checkpoint is saved
RESUME_FROM = "checkpoints/minipile_ckpt_step_21600.pt"      # The checkpoint file that the training will resume on in case of an interuption
DATA_FILE = "../data/minipile.bin"
CHECKPOINT_DIR = "../checkpoints"


def cross_entropy(logits, targets):
    """
    Args:
        logits: The tensor of shape (B, T, VOCAB_SIZE) that represents the final unnormalized scores of all the words
        targets: The matrix of shape (B, T) that represents the correct word inside the dataset

    Returns:
        A scalar value that is the average loss of the logits

    Logic:
        1. Same softmax stability trick used in my implementation of attention, subtract the max along the VOCAB_SIZE axis because e to the power of a massive number could result in integer overflow. So we horizontally shift the scores towards negatives because there's a lot less e^x variance there
        2. Calculate the softmax, normally it's a fraction inside a log function, but we can use properties to turn it into a subtraction. Now instead of unnormalized scores, the components are probabilities
        3. Initialize indices for pytorch to use for comparison, retrieve B * T for total number of tokens in logits
        4. Use integer array indexing (coordinate lookup basically) to create a vector of probabilities for the correct token in the logits
        5. Mean the correct token probabilities vector to collapse the list into a scalar (our optimizer expects a scalar loss value), then make it negative because the optimizer tries to minimize loss, then by minimizing a negative value you're maximizing the positive component. Meaning we "trick" our optimizer into maximizing the probabilities for the correct token. This is called Negative Log Likelihood
    """

    logits = logits - logits.max(dim=-1, keepdim=True)[0]
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    B_T, V = logits.shape
    row_indices = torch.arange(B_T, device=DEVICE)
    correct_log_probs = log_probs[row_indices, targets]
    return -correct_log_probs.mean()


class DataLoader:
    """
    A middleman between the GPU and dataset, it chops up the dataset into BLOCK_SIZE batches, prepares a matrix of shape (PHYSICAL_BATCH_SIZE, BLOCK_SIZE) to give to the GPU, then it prepares sets of input and targets
    """

    def __init__(self, filename, block_size, batch_size):
        """
        Args:
            filename: The dataset file
            block_size: How long a sequence is in tokens
            batch_size: How many sequences are going to be processed in parallel

        Logic:
            1. Initiate block size, batch size, open file, tokenize, store
            2. We initialize tokens with torch.long because the encodings are massive, and device is cpu because cpu memory is cheap and throwaway, perfect for storing the tokens
        """

        self.block_size = block_size
        self.batch_size = batch_size

        self.tokens = np.memmap(filename, dtype=np.uint16, mode='r')

        self.current_position = 0

    def get_batch(self):
        """
        Returns:
            The input and targets for one batch

        Logic:
            1. Initialize Batch and Time
            2. If the next batch goes past the dataset, the integer current_position will reset to 0 to simulate an epoch
            3. The buffer is our batch of tokens that are grabbed as well as 1 extra token to serve as the target
            4. We shift the x and y to create the input target pair. the same index on y will provide the very next token in the original sequence for x, then we reshape the long vector into (B, T) for parallel processing later
            5. The current position is then jumped forward by the batch to repeat on the next batch
        """

        B, T = self.batch_size, self.block_size

        if self.current_position + (B * T) + 1 > len(self.tokens):
            self.current_position = 0

        buf = self.tokens[self.current_position: self.current_position + (B * T) + 1]

        x = torch.from_numpy(buf[:-1].astype(np.int64)).view(B, T)
        y = torch.from_numpy(buf[1:].astype(np.int64)).view(B, T)

        self.current_position += B * T

        return x.to(DEVICE), y.to(DEVICE)


def get_lr(it):
    """
    The learning rate scheduler (we use cosine decay)

    Args:
        it: current iteration count

    Returns:
        The learning rate to use

    Logic:
        1. First warm up the model by initializing learning rate (lr) to 0, since at the start the error gradients are massive, and we don't want the step size to jump the weights around as much (prevent overshoots)
        2. If training goes for longer than expected, use the minimum lr
        3. The decay ratio takes the current iteration and calculates a number between 0 and 1 that of how far you are in training. We do this because using the raw iteration to stretch out a portion of cosine's curve to use over the entire session
        4. Use cosine to calculate a start-high end-low curve for our learning rate
    """

    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / (WARMUP_ITERS + 1)

    if it > MAX_ITERS:
        return MIN_LR

    decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


if __name__ == "__main__":
    """
    Logic:
        1. Set the device to mps so tensors are made on the GPU by default, set a manual seed for reproducibility (since it's training)
        2. Initialize everything, we use AdamW (Adaptive Moment Estimation with Weight Decay) for the optimizer. Instead of a global learning rate for all weights, AdamW increases the learning rate for a weight if it's been steadily going the same direction, and lowers it if theres a lot of bumps or direction changes. betas represent the momentum memory, how far AdamW looks back, and weight decay is for making the weights smaller artificially each update to help prevent overfitting
        3. A checkpoint loader, in case of any interruptions. Loads the weights and AdamW's momentum at that step
        4. Put the model in training mode (enables dropout)
    """

    torch.set_default_device(DEVICE)
    torch.manual_seed(1337)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader = DataLoader(DATA_FILE, BLOCK_SIZE, PHYSICAL_BATCH_SIZE)
    model = GPT()
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)

    start_iter = 0

    if RESUME_FROM:
        print(f"using checkpoint: {RESUME_FROM}")
        checkpoint = torch.load(RESUME_FROM, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']

    model.train()
    start_time = time.time()
    loss_accum = 0.0

    try:
        for iter in range(start_iter, MAX_ITERS):
            """
            Logic: 
                1. In each iteration, get the learning rate and use it, delete old gradients for memory effeciency
                2. Accumulate the losses from each micro-step
                3. Ensure the gradients are not massive by clipping it (should be somewhat prevented with warmup but just in case)
                4. Take the step for each weight, updating them
                5. Logging and checkpointing
            """

            lr = get_lr(iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0

            for micro_step in range(GRAD_ACCUM_STEPS):
                """
                Logic:
                    1. Get logits (predictions) from model, then pass the last axis of predictions and targets to cross entropy for loss
                    2. Because we are adding the losses of many micro batches together before updating, we need to divide the loss by that amount to make sure the value stays correct when we update weights
                    3. Add the loss to the accumulated loss
                    4. Store the gradients
                """

                x, y = train_loader.get_batch()
                logits = model(x)

                loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / GRAD_ACCUM_STEPS
                loss_accum += loss.detach()
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if iter % 10 == 0:
                dt = time.time() - start_time
                tokens_processed = (PHYSICAL_BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE) * 10
                speed = tokens_processed / (dt + 1e-6)

                print(f"Step {iter}: Loss: {loss_accum:.4f}, LR: {lr:.2e}, Speed: {speed:.0f} tok/s")

                with open("../logs/minipile_training_log.csv", "a") as f:
                    f.write(f"{iter},{loss_accum:.4f}\n")
                start_time = time.time()

            if iter > 0 and iter % CHECKPOINT_INTERVAL == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"minipile_ckpt_step_{iter}.pt")

                print(f"saved checkpoint: {ckpt_path}")

                torch.save({
                    'step': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_accum,
                }, ckpt_path)

    except KeyboardInterrupt:
        torch.save({
            'step': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_accum,
        }, os.path.join(CHECKPOINT_DIR, "interrupted.pt"))

        exit(0)

    torch.save(model.state_dict(), "minipile.pt")