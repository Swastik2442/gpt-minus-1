import torch
import torch.nn as nn
import torch.optim as optim

class GPTMinus1(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, dropout=0.1):
        super(GPTMinus1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) # Not BPE
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_length = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, d_model)
        logits = self.fc_out(x)
        return logits

def create_optimizer(model, lr=1e-4, weight_decay=0.01):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def create_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

def save_model(model, optimizer, scheduler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)

def load_model(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def train_step(model, optimizer, scheduler, criterion, inputs, targets):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

def eval_step(model, criterion, inputs, targets):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    return loss.item()

def generate_text(model, start_token, max_length, tokenizer, device):
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
    # Example usage
    vocab_size = 30522  # Example vocab size for BERT tokenizer
    d_model = 768
    n_heads = 12
    n_layers = 6
    model = GPTMinus1(vocab_size, d_model, n_heads, n_layers)

    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, num_warmup_steps=1000, num_training_steps=10000)
    criterion = nn.CrossEntropyLoss()

    # Dummy data for demonstration
    inputs = torch.randint(0, vocab_size, (8, 128))  # (batch_size, seq_length)
    targets = torch.randint(0, vocab_size, (8, 128))

    loss = train_step(model, optimizer, scheduler, criterion, inputs, targets)
    print(f"Training loss: {loss}")

    save_model(model, optimizer, scheduler, epoch=1, path="model_checkpoint.pth")

    eval_loss = eval_step(model, criterion, inputs, targets)
    print(f"Evaluation loss: {eval_loss}")
