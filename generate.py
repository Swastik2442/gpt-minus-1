import torch
import tiktoken
# from tokenizers import Tokenizer

from config import Config
from model import GPTMinus1

# tokenizer = Tokenizer.from_file("../hi_bpe_tokenizer.json")
tokenizer = tiktoken.get_encoding("gpt2")

def generate_text(model: GPTMinus1, config: Config, start_tokens: str, max_length: int):
    model.eval()
    # generated = tokenizer.encode(start_tokens).ids
    generated = tokenizer.encode(start_tokens)
    gen_length = len(generated)

    for _ in range(max_length - gen_length):
        with torch.no_grad():
            generated = generated[-config.n_ctx:]
            input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(config.device)
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            print(tokenizer.decode([next_token]), end='') # type: ignore
            generated.append(next_token) # type: ignore
    print()

    return tokenizer.decode(generated)

if __name__ == "__main__":
    cfg = Config.load()

    model = GPTMinus1(
        cfg.vocab_size,
        cfg.n_ctx,
        cfg.d_model,
        cfg.n_heads,
        cfg.n_layers,
        device=cfg.device
    )
    model.load("out/model_checkpoint_1000.pth")

    generate_text(model, cfg, "\n", 512)
