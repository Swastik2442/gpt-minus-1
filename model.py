import math
import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k_sqrt_inv = 1.0 / math.sqrt(self.d_model // self.n_heads)

        # self.q = torch.nn.Linear(self.d_model, self.d_model)
        # self.k = torch.nn.Linear(self.d_model, self.d_model)
        # self.v = torch.nn.Linear(self.d_model, self.d_model)
        self.qkv = torch.nn.Linear(self.d_model, 3 * self.d_model)
        self.proj = torch.nn.Linear(self.d_model, self.d_model)
        self.softmax = torch.nn.Softmax(-1)

        self.attn_dropout = torch.nn.Dropout(dropout)
        self.proj_dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, C, D = x.size() # batch size, n_ctx, d_model

        # q = self.q(x)
        # k = self.k(x)
        # v = self.v(x)
        q, k, v = self.qkv(x).split(self.d_model, dim=2)

        # move head forward to be the batch dimension
        q = q.view(B, C, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, nh, C, hs)
        k = k.view(B, C, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, nh, C, hs)
        v = v.view(B, C, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, nh, C, hs)

        # calculate self-attention
        att = (q @ k.transpose(-2, -1)) * self.d_k_sqrt_inv # (B, nh, C, hs) x (B, nh, hs, C) -> (B, nh, C, C)
        att = self.softmax(att)
        att = self.attn_dropout(att)
        y = att @ v                                         # (B, nh, C, C) x (B, nh, C, hs) -> (B, nh, C, hs)
        y = y.transpose(1, 2).contiguous().view(B, C, D)    # re-assemble

        y = self.proj(y)
        y = self.proj_dropout(y)
        return y

class FFNN(torch.nn.Module):
    def __init__(self, d_model: int, dropout=0.1):
        super().__init__()
        self.layer1 = torch.nn.Linear(d_model, 4 * d_model)
        self.layer2 = torch.nn.Linear(4 * d_model, d_model)
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        # x = self.gelu(x)
        x = self.dropout(x)
        return x

class Block(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln_1 = torch.nn.LayerNorm(d_model)
        self.ffnn = FFNN(d_model, dropout)
        self.ln_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # Post
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.ffnn(x))

        # # Pre
        # x = x + self.attn(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))
        return x

class GPTMinus1(torch.nn.Module):
    def __init__(self, vocab_size: int, n_ctx: int, d_model: int, n_heads: int, n_layers: int, dropout=0.1):
        super(GPTMinus1, self).__init__()
        self.text_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_encoding = torch.nn.Embedding(n_ctx, d_model)

        self.transformer = torch.nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])

        self.linear = torch.nn.Linear(d_model, vocab_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        seq_length = x.size(1)
        x = self.text_embedding(x) + self.positional_encoding(torch.arange(0, seq_length, dtype=torch.long))
        x = self.dropout(x)
        for block in self.transformer:
            x = block(x)
        x = self.linear(x)
        return x

def create_optimizer(model: GPTMinus1, lr: float = 1e-4, weight_decay: float = 0.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def create_scheduler(optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

def save_model(model: GPTMinus1, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, path: str):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)

def load_model(model: GPTMinus1, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, path: str):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def train_step(model: GPTMinus1, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, criterion: torch.nn.CrossEntropyLoss, inputs: torch.Tensor, targets: torch.Tensor):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

def eval_step(model: GPTMinus1, criterion: torch.nn.CrossEntropyLoss, inputs: torch.Tensor, targets: torch.Tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    return loss.item()

def generate_text(model: GPTMinus1, start_token, max_length, tokenizer, device):
    model.eval()
    generated = [start_token]
    input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_length - 1):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            generated.append(next_token)
            input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)

    return tokenizer.decode(generated)

if __name__ == "__main__":
    batch_size = 8
    vocab_size = 1024
    n_ctx = 256
    d_model = 256
    n_heads = 4
    n_layers = 2
    epochs = 1000
    model = GPTMinus1(vocab_size, n_ctx, d_model, n_heads, n_layers)

    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, num_warmup_steps=1000, num_training_steps=10000)
    criterion = torch.nn.CrossEntropyLoss()

    # Dummy data for demonstration
    inputs = torch.randint(0, vocab_size, (batch_size, n_ctx, d_model))
    targets = torch.randint(0, vocab_size, (batch_size, n_ctx, d_model))

    epoch_data = []
    for epoch in range(1, epochs+1):
        loss = train_step(model, optimizer, scheduler, criterion, inputs, targets)
        eval_loss = eval_step(model, criterion, inputs, targets)

        epoch_data.append({"epoch": epoch, "training_loss": loss, "eval_loss": eval_loss})
        print(f"Epoch {epoch} | Training loss: {loss} | Evaluation loss: {eval_loss}")

        save_model(model, optimizer, scheduler, epoch, f"model_checkpoint_{epoch}.pth")
