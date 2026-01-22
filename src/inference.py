import torch
import tiktoken
from model import GPT, DEVICE, BLOCK_SIZE

CHECKPOINT = "checkpoints/interrupted.pt"
enc = tiktoken.get_encoding("gpt2")


def encode(text):
    """
    Args:
        text: Input string

    Returns:
        A tensor of integers representing the string

    Logic:
        1. Tiktoken breaks the string down into sub-word chunks with a machine learning model, using character pair counts and stuff, then corresponds each sub-word to its integer token
        2. We take those integer tokens and store them in a tensor object
    """

    return torch.tensor(enc.encode(text), dtype=torch.long)


def decode(idxs):
    """
    Args:
        idxs: A tensor of integers (token id)

    Returns:
        A string represented by the integers

    Logic:
        1. Convert the tensor to a list
        2. Tiktoken does a reverse process I believe. I don't really know how the decoder works
    """

    if isinstance(idxs, torch.Tensor):
        idxs = idxs.tolist()

    return enc.decode(idxs)


def generate(model, idx, max_new_tokens, temperature, top_k, eos_id, repetition_penalty):
    """
    Args:
        model: The GPT object, that takes input token tensor and outputs a score for each token tensor
        idx: Tensor of integer tokens
        max_new_tokens: The maximum amount of new tokens the model is allowed to generate (we get to control this because GPT is autoregressive, meaning it generates one token at a time)
        temperature: A creativity control, higher temperate flattens the probability distribution, making a lot of tokens equally likely. Lower temperature sharpens the distribution, making the already likely token even more likely
        top_k: Only pay attention to the top "k" likely tokens, ignoring the rest
        eos_id: The token ID for the end of sequence token, used to stop generation after the model is done playing the role of the "assistant"

    Returns:
        A tensor of each token generated after every autoregressive step

    Logic:
        1. Disable dropout (since it's inference)
        2. Forward pass, disabling gradient calculation since we are not training
        3. Crop out the time/context length axis except the last token to use it for generation: (B, T, VOCAB_SIZE) -> (B, VOCAB_SIZE)
        5. Reduce repetition by lowering probabilities of tokens in the logits that are already said
        4. Apply the temperature scaling
        5. Find the top_k probability cutoff, then apply it by setting the score of every token who's below the cutoff to -inf (softmax makes it 0)
        6. Perform softmax for probabilities of every token
        7. Go through the probability distribution and randomly pick a token
        8. If the chosen token is the end of sequence token, end the loop. Otherwise, add the chosen token to the list of output tokens
    """
    
    model.eval()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(idx)

        logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            for i in range(idx.shape[0]):
                for previous_token in set(idx[i].tolist()):
                    if logits[i, previous_token] < 0:
                        logits[i, previous_token] *= repetition_penalty
                    else:
                        logits[i, previous_token] /= repetition_penalty

        logits = logits / temperature

        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")

        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        probs = logits.exp()
        probs = probs / probs.sum(dim=-1, keepdim=True)

        idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=-1)

    return idx


if __name__ == "__main__":
    """
    Logic:
        1. Initialize the model and move all parameters to the V-RAM
        2. Start interaction loop
        3. Tokenize the prompt, then add a "Batch Dimension" because the model expects it: (T) -> (B, T), then move it to the V-RAM
        4. Generate the token ids, then pass them to the decoder, index the token tensor because we want the first batch's token ids
        5. Sliding context window
    """

    model = GPT()
    model.to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE)['model_state_dict'])

    system_prompt = (
        "The following is a chat with an AI assistant who is helpful, smart, and brief.\n"
        "User: Hi.\n"
        "Assistant: Hey! What's up?\n"
        "User: Who are you?\n"
        "Assistant: I'm AI epstn. Ask me anything.\n"
        "User: What is the sun?\n"
        "Assistant: It's the star at the center of our solar system.\n"
    )

    history_ids = enc.encode(system_prompt)

    while True:
        user_input = input("User: ")

        if user_input == "q":
            break

        new_input_text = f"User: {user_input}\nAssistant (briefly):"
        new_ids = enc.encode(new_input_text)
        history_ids.extend(new_ids)

        max_new_tokens = 100
        safe_buffer = 10

        if len(history_ids) + max_new_tokens + safe_buffer > BLOCK_SIZE:
            sys_len = len(enc.encode(system_prompt))
            available_space = BLOCK_SIZE - sys_len - 150

            if len(history_ids) > (BLOCK_SIZE - 150):
                history_ids = history_ids[:sys_len] + history_ids[-available_space:]

        input_tensor = torch.tensor(history_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

        generated_ids_tensor = generate(model, input_tensor, 100, 0.6, 40, 50256, 1.2)

        full_sequence = generated_ids_tensor[0].tolist()
        new_response_ids = full_sequence[len(history_ids):]

        output_text = enc.decode(new_response_ids)
        if "User:" in output_text:
            output_text = output_text.split("User:")[0]
        print(f"Model: {output_text.strip()}")

        history_ids.extend(new_response_ids)