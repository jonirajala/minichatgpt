import torch
import os
import tiktoken
from model import LLama
from pretrain import Config


def load_models(parameter_count, enc, directory='trained_models'):
    models = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            model_name = filename
            model_path = os.path.join(directory, filename)
            weights = torch.load(model_path, map_location=torch.device('cpu'))
            config_dict = weights['config']
            config = Config(enc.n_vocab)
            config.__dict__.update(config_dict)
            model = LLama(config)
            model.load_state_dict(weights['model_state_dict'])
            models[model_name] = model
    
    return models

def generate_text_with_models(models, enc,  device='mps'):
    for model_name, model in models.items():
        print(f"----Model: {model_name}----")
        model = model.to(device)
        model.eval()
        inp = torch.tensor(enc.encode("Hallitus oli ")).to(device)
        with torch.no_grad():
            gen_text = model.generate(inp).detach().cpu().numpy()
        gen_text = enc.decode(gen_text)
        print(gen_text, "\n")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device =  "mps"
    else:
        device = "cpu"

    enc = tiktoken.get_encoding("gpt2")
    parameter_count = 100
    print(f"Searching {parameter_count}M models")
    models = load_models(parameter_count, enc)
    print(f"loaded {len(models)} models")
    generate_text_with_models(models, enc, device)
