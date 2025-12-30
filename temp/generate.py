import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import configuration
from model import Transformer

# --- 1. Setup ---
device = torch.device(configuration.device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = Transformer().to(device)

# --- 2. Load the Weights ---
# CHANGE THIS to your latest file (e.g., model_epoch_5.pth) to see the best results!
model_path = "model_epoch_5.pth"

print(f"Loading model weights from {model_path}...")
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Weights loaded successfully!")
except FileNotFoundError:
    print(f"Error: Could not find {model_path}. Did you train it yet?")
    exit()

model.eval()


# --- 3. The Top-K Generation Function ---
def generate_response(prompt_text, max_new_tokens=50, temperature=0.8, top_k=5):
    # Prepare the prompt
    full_prompt = f"User: {prompt_text}\nAssistant:"
    input_ids = tokenizer.encode(full_prompt, return_tensors='pt').to(device)

    for _ in range(max_new_tokens):
        # Crop context if it gets too long
        input_condensed = input_ids[:, -configuration.block_size:]

        with torch.no_grad():
            logits = model(input_condensed)

        # Focus on the last token's prediction
        logits = logits[:, -1, :]

        # --- THE MAGIC SAUCE (Top-K Sampling) ---
        # 1. Apply Temperature (Makes the distribution flatter or sharper)
        # Higher temp (1.0+) = more random/creative
        # Lower temp (<1.0) = more confident/conservative
        logits = logits / temperature

        # 2. Filter to Top-K (Keep only the K best options)
        # We set everything else to negative infinity so they are never picked
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')

        # 3. Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # 4. Sample from the distribution
        # Instead of just picking the #1 best, we roll a weighted die
        predicted_token_id = torch.multinomial(probs, num_samples=1)

        # Append and Repeat
        input_ids = torch.cat((input_ids, predicted_token_id), dim=1)

        if predicted_token_id.item() == tokenizer.eos_token_id:
            break

    decoded_output = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Optional: Clean up the output to only show the Assistant's part
    # return decoded_output.split("Assistant:")[-1].strip()
    return decoded_output


# --- 4. Chat Loop ---
print("\n--- Model Ready! (Type 'quit' to exit) ---")
print(f"Loaded: {model_path}")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # Play with temperature! 0.7 is usually a sweet spot for chat.
    response = generate_response(user_input, temperature=0.6, top_k=10)
    print(f"\n{response}\n")
    print("-" * 20)