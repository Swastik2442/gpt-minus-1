import torch
import tiktoken
# from tokenizers import Tokenizer

from config import Config
from model import GPTMinus1

# tokenizer = Tokenizer.from_file("../hi_bpe_tokenizer.json")
tokenizer = tiktoken.get_encoding("gpt2")

def generate_text(
    model: GPTMinus1,
    config: Config,
    start_tokens: str,
    max_length: int,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float = 1.0
):
    assert top_k is None or top_k > 0
    assert top_p is None or (top_p > 0.0 and top_p <= 1.0)
    assert temperature > 0.0

    model.eval()
    # generated = tokenizer.encode(start_tokens).ids
    generated = tokenizer.encode(start_tokens)

    for _ in range(max_length):
        with torch.no_grad():
            inputs = torch.tensor(generated[-config.n_ctx:], dtype=torch.long, device=config.device).unsqueeze(0)

            logits: torch.Tensor = model(inputs)
            logits = logits[0, -1, :] / temperature # first batch, last token, all probabilities

            # Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
            if top_k is not None: # take only top k logits
                top_k = min(top_k, logits.size(-1))
                val, _idx = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < val[-1], float('-inf'))
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p

                # Shift the indices to the right once to keep at least the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, -1)
            next_token = torch.multinomial(probs, num_samples=1)[0].item()
            # yield tokenizer.decode([next_token]) # type: ignore
            generated.append(next_token) # type: ignore

    return tokenizer.decode(generated)

if __name__ == "__main__":
    import os
    import sys

    ckpt_path = "out/shakespeare/model_checkpoint_1000.pth"
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint path '{ckpt_path}' does not exist.")
            sys.exit(1)

    print("Loading Config...")
    cfg = Config.load(os.path.join(os.path.dirname(ckpt_path), "config.json"))

    print("Loading Model...")
    model = GPTMinus1(
        cfg.vocab_size,
        cfg.n_ctx,
        cfg.d_model,
        cfg.n_heads,
        cfg.n_layers,
        device=cfg.device
    )
    model.load(ckpt_path)

    print("Generating Text...\n")
    print(generate_text(model, cfg, "\n", 512))
