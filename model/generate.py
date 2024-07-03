import torch
import os
import tiktoken
from model.CAM_model import CAM
from pretrain import Config
from torch.quantization import get_default_qconfig, float_qparams_weight_only_qconfig, prepare, convert

# torch compile adds _orig_mod.
def correct_dict_keys(wrong_dict):
    prefix_to_remove = '_orig_mod.'
    # Create a new dictionary with corrected keys
    corrected_dict = {key[len(prefix_to_remove):] if key.startswith(prefix_to_remove) else key: value 
                      for key, value in wrong_dict.items()}
    
    return corrected_dict

def apply_qconfig_to_embedding_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.qconfig = float_qparams_weight_only_qconfig
        else:
            module.qconfig = get_default_qconfig('qnnpack')


def load_models(enc, directory='trained_models'):
    models = {}

    
    
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            torch.backends.quantized.engine = 'qnnpack'

            model_name = filename
            model_path = os.path.join(directory, filename)
            weights = torch.load(model_path, map_location=torch.device('cpu'))
            config_dict = weights['config']
            config = Config(enc.n_vocab)
            config.__dict__.update(config_dict)
            weights['model_state_dict'] = correct_dict_keys(weights['model_state_dict'])
            model = weights['model']

            model.train()
            model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
            model_fp32_prepared = torch.quantization.prepare_qat(model)
            apply_qconfig_to_embedding_layers(model_fp32_prepared)
            model_int8 = torch.quantization.convert(model_fp32_prepared)

            model_int8.load_state_dict(weights['model_state_dict'])

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

    directory = "quantized_models"

    enc = tiktoken.get_encoding("gpt2")
    models = load_models(enc, directory)
    print(f"loaded {len(models)} models")
    generate_text_with_models(models, enc, device)
