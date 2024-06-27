import numpy as np
import torch
import tiktoken
import os
from torch import optim
from tqdm.auto import tqdm
from model import LLama
import math
import inspect
import json
from datetime import datetime


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

warmup_steps = 10
max_step = 50
max_lr = 6e-4
min_lr = 0.1 * max_lr
def get_lr(step):
    if step < warmup_steps:
        return(step+1)/warmup_steps * max_lr
    elif step < max_step:
        decay_ratio = (step-warmup_steps) / (max_step-warmup_steps)
        coeff = 0.5 * (1.0+math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr-min_lr)
    else:
        return min_lr

def save_losses(train_losses, val_losses, config):
    f_name = f"losses/{config.iters}_{datetime.now().strftime('%d-%m')}.json"
    
    if os.path.exists(f_name):
        with open(f_name, "r") as f:
            losses = json.load(f)
    else:
        losses = {"train_losses": [], "val_losses": []}
    
    losses["train_losses"].extend(train_losses)
    losses["val_losses"].extend(val_losses)
    
    with open(f_name, "w") as f:
        json.dump(losses, f)
    print(f"Losses saved to {f_name}")

def save_model(model, model_name, step):
    model_save_path = os.path.join("trained_models", f"{model_name}_{step}step.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

class DataLoader:
    def __init__(self, data, batch_size, block_size):
        self.data = data

        self.batch_size = batch_size
        self.block_size = block_size
        self.pos = 0

    def get_batch(self):
        B, T = self.batch_size, self.block_size
        batch = self.data[self.pos : self.pos + B * T + 1]
        x = torch.tensor(batch[:-1], dtype=torch.long).reshape(B, T).to(device)
        y = torch.tensor(batch[1:], dtype=torch.long).reshape(B, T).to(device)
        self.pos += B * T

        if self.pos + (B * T + 1) > len(self.data):
            self.pos = 0
        return x, y

class Config:
    def __init__(self, vocab_size):
        self.block_size = 96
        self.batch_size = 32
        self.iters = 1000
        self.dropout = 0.1

        self.emb_dim = 768
        self.n_layers = 12
        self.n_head = 12
        self.flash = True
        self.vocab_size = vocab_size

if __name__ == "__main__":
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs("losses", exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device =  "mps"

    print(f"Training on {device} device")

    train_data = np.array(
        np.memmap(
            os.path.join("data", "finnish_train.bin"), dtype=np.uint16, mode="r"
        )
    )
    val_data = np.array(
        np.memmap(os.path.join("data", "finnish_val.bin"), dtype=np.uint16, mode="r")
    )

    model_name = "CAL" # crazyassllm
    config = Config(enc.n_vocab)
    model = LLama(config).to(device)
    
    if device == 'cuda':
        model = torch.compile(model)

    params = count_parameters(model)
    model_name = f"{model_name}-{params // 1_000_000}M"

    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device == 'cude'
    optimizer = optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

    print(
        f"Training on {(config.batch_size * config.iters * config.block_size) // 1_000_000}M tokens"
    )
    print(f"Model: {model_name:<10} | Params: {params:>10,}")

    trainloader = DataLoader(train_data, config.batch_size, config.block_size)
    valloader = DataLoader(val_data, config.batch_size, config.block_size)

    train_losses = []
    val_losses = []
    model.train()
    pbar = tqdm(range(config.iters), desc="Training Progress")
    for step in pbar:
        x, y = trainloader.get_batch()
        model.zero_grad()
        if device == 'cuda':
            # currently google colabs t4 gpu doesnt support compile + bfloat16
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=False):
                out, loss = model(x, y)
        else:
            out, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        pbar.set_postfix({"train_loss": loss.item()})
        train_losses.append(loss.item())

        if step % 50 == 0:
            model.eval()
            val_x, val_y = valloader.get_batch()
            with torch.no_grad():
                val_out, val_loss = model(val_x, val_y)
            val_losses.append(val_loss.item())
            model.train()

        if step % 500 == 0:
            save_model(model, model_name, step)
            save_losses(train_losses, val_losses, config)
            train_losses = []
            val_losses = []
            print(f"Model and losses saved to on step {step}")
    
    save_model(model, model_name, step)
    save_losses(train_losses, val_losses, config)

    model.eval()
    inp = torch.tensor(enc.encode("And that is  ")).to(device)
    gen_text = model.generate(inp).detach().cpu().numpy()
    gen_text = enc.decode(gen_text)
    print(gen_text)
 