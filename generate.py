import tiktoken
import torch

from model import GPTMinus1, create_scheduler, create_optimizer, load_model

batch_size = 8
vocab_size = 50257 # GPT-2 Vocab
n_ctx = 256
d_model = 256
n_heads = 4
n_layers = 2
num_warmup_steps = 4000
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = tiktoken.get_encoding("gpt2")

def generate_text(model: GPTMinus1, start_token: int, max_length: int):
    model.eval()
    generated = [start_token]
    input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_length - 1):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            generated.append(next_token) # type: ignore
            input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)

    return tokenizer.decode(generated)

if __name__ == "__main__":
    model = GPTMinus1(vocab_size, n_ctx, d_model, n_heads, n_layers, device=device).to(device)

    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, d_model, num_warmup_steps)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    load_model(model, optimizer, scheduler, "out/model_checkpoint_1000.pth")
    print(generate_text(model, tokenizer.encode("\n")[0], n_ctx))
