from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import os
import tiktoken

from model.model import LLama
from model.pretrain import Config

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.json.ensure_ascii = False
app.json.mimetype = "application/json; charset=utf-8"

enc = tiktoken.get_encoding("gpt2")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

model_path = os.path.join("../model/trained_models/CAL-76M_4M_TOKENS.pt")
weights = torch.load(model_path, map_location=torch.device('cpu'))
config_dict = weights['config']
config = Config(enc.n_vocab)
config.__dict__.update(config_dict)
model = LLama(config)
model.load_state_dict(weights['model_state_dict'])
model.to(device)
model.eval()

@app.route('/response', methods=['GET'])
def get_response():
    prompt = request.args.get('prompt')
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    prompt_tensor = torch.tensor(enc.encode(prompt)).unsqueeze(0).to(device)
    with torch.no_grad():
        generated_ids = model.generate(prompt_tensor).squeeze(0).detach().cpu().numpy()
    
    gen_text = enc.decode(generated_ids.tolist())
    return jsonify({"response": gen_text}), 200, {'Content-Type': 'application/json; charset=utf-8'}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
