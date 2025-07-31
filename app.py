from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
from model import TextGenModel
from utils import load_tokenizer

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.route("/script.js")
def script():
    with open("script.js", "r", encoding="utf-8") as f:
        return f.read()

tokenizer = load_tokenizer()
model = TextGenModel(tokenizer.vocab_size())
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt", "")
    max_len = 50

    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_len):
            output, _ = model(input_ids)
            next_token = output[0, -1].argmax().item()
            tokens.append(next_token)
            input_ids = torch.tensor(tokens).unsqueeze(0)

    return jsonify({"generated": tokenizer.decode(tokens)})

if __name__ == "__main__":
    app.run(port=5000)
