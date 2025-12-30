import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
import configuration
from model import Transformer


device = torch.device(configuration.device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("blended_skill_talk", split="train")


def format_and_tokenize(examples):
    outputs = []

    for i in range(len(examples["free_messages"])):
        user_message = examples["previous_utterance"][i] if len(examples["previous_utterance"][i]) > 0 else "Hello"
        assist_message = examples["free_messages"][i][0] if len(examples["free_messages"][i]) > 0 else "Hi there"

        conversation = f"User: {user_message}\nAssistant: {assist_message}{tokenizer.eos_token}"
        outputs.append(conversation)

    return tokenizer(
        outputs,
        padding="max_length",
        truncation=True,
        max_length=configuration.block_size,
        return_tensors="pt",
    )


tokenized_dataset = dataset.map(format_and_tokenize, batched=True, remove_columns=dataset.column_names)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

train_loader = DataLoader(tokenized_dataset, batch_size=configuration.batch_size, shuffle=True)

model = Transformer().to(device)
optimizer = AdamW(model.parameters(), lr=configuration.learning_rate)

model.train()

for epoch in range(configuration.epochs):
    loop = tqdm(train_loader, leave=True)
    total_loss = 0

    for batch in loop:
        input_ids = batch["input_ids"].to(device)

        x = input_ids[:, :-1]
        y = input_ids[:, 1:]

        optimizer.zero_grad()
        logits = model(x)

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, configuration.vocabulary_size),
            y.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_description(f"epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} finished. Avg Loss: {average_loss:.4f}")

    save_path = f"model_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), save_path)

print(f"Saved model to {save_path}")