import torch
from torch.quantization import get_default_qconfig, float_qparams_weight_only_qconfig, prepare, convert
import os
import numpy as np
from pretrain import Config
import tiktoken
from tqdm.auto import tqdm
from model.CAM_model import CAM
print("Available quantized backends:", torch.backends.quantized.supported_engines)

torch.backends.quantized.engine = 'qnnpack'
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


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# not working on mps for some reason
# elif torch.backends.mps.is_available():
#     device =  "mps"


class DataLoader:
    def __init__(self, data, batch_size, block_size):
        self.data = data
        print(f"Loaded {len(data)} tokens")

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
    

os.makedirs("quantized_models", exist_ok=True)

calib_data = np.array(
    np.memmap(os.path.join("data","finnish_val.bin"), dtype=np.uint16, mode="r")
)
enc = tiktoken.get_encoding("gpt2")

model_name = "CAL-304M_8M_TOKENS.pt"
model_path = os.path.join("trained_models", model_name)
weights = torch.load(model_path, map_location=torch.device('cpu'))
config_dict = weights['config']
config = Config(enc.n_vocab)
config.__dict__.update(config_dict)
weights['model_state_dict'] = correct_dict_keys(weights['model_state_dict'])
model = CAM(config)
model.load_state_dict(weights['model_state_dict'])

model.to(device)
import copy
og_model = copy.deepcopy(model)
model.eval()  # Set the model to evaluation mode

# Specify quantization configuration
apply_qconfig_to_embedding_layers(model)

# Prepare the model for quantization
prepare(model, inplace=True)

trainloader = DataLoader(calib_data, config.batch_size, config.block_size)

# Calibrate the model (using a DataLoader with representative data)
# Uncomment and customize the following lines as needed
pbar = tqdm(range(2), desc="Training Progress", dynamic_ncols=True)
for step in pbar:
    x, y = trainloader.get_batch()
    out, _ = model(x, 0)

# Convert the model to quantized version
quantized_model = convert(model, inplace=True)

quantized_model_save_path = os.path.join("quantized_models", f"quant_{model_name}")
torch.save({
    'model_state_dict': quantized_model.state_dict(),
    'model': quantized_model,
    'config': config.__dict__
}, quantized_model_save_path)

print(f"Quantized model and config saved to {quantized_model_save_path}")


# https://discuss.pytorch.org/t/loading-of-quantized-model/64213/3

"""
Make sure you create the net using previous definition, and let the net go through process that was applied during quantization before (prepare_model, fuse_model, and convert), without rerun the calibration process.
After that you can load the quantized state_dict in. Hope it helps.
"""