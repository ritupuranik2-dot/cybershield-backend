from __future__ import annotations
import os, re, unicodedata
from pathlib import Path
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoConfig, AutoModel, AutoTokenizer

app = Flask(__name__)
CORS(app, origins="*")

DEVICE     = torch.device("cpu")
MODEL_DIR  = Path("saved_model")
HF_REPO    = os.environ.get("HF_REPO", "")
MAX_LENGTH = 128

L1_NAMES = {0: "safe", 1: "threat"}
L2_NAMES = {0: "harassment", 1: "violent", 2: "rape"}

class HierarchicalThreatClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.3):
        super().__init__()
        config       = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden       = config.hidden_size
        self.head1   = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 2),
        )
        self.head2   = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 3),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out    = self.encoder(**kwargs)
        pooled = out.last_hidden_state[:, 0, :]
        return self.head1(pooled).float(), self.head2(pooled).float()

model     = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is not None:
        return
    if not MODEL_DIR.exists() and HF_REPO:
        print(f"Downloading model: {HF_REPO}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=HF_REPO, local_dir=str(MODEL_DIR))
    if not MODEL_DIR.exists():
        raise RuntimeError("Model not found. Set HF_REPO env var.")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    print("Loading model weights...")
    model = HierarchicalThreatClassifier(str(MODEL_DIR))
    heads = torch.load(MODEL_DIR / "heads.pt", map_location=DEVICE)
    model.head1.load_state_dict(heads["head1"])
    model.head2.load_state_dict(heads["head2"])
    model.to(DEVICE).eval()
    print("Model ready!")

AGGRESSIVE_EMOJIS     = set("😡🤬👊🔪💀☠️🔫💣⚔️🖕🩸")
SEXUAL_EMOJIS         = set("🍆🍑😈💦🔞")
EMOJI_CATEGORY_TOKENS = {"aggressive": "[EMOJI_AGGR]", "sexual": "[EMOJI_SEX]", "general": "[EMOJI_GEN]"}

def augment_with_emoji_prefix(text: str) -> str:
    seen: dict[str, bool] = {}
    for ch in text:
        if ch in AGGRESSIVE_EMOJIS:      seen["aggressive"] = True
        elif ch in SEXUAL_EMOJIS:        seen["sexual"]     = True
        else:
            cp = ord(ch); cat = unicodedata.category(ch)
            if cat in ("So","Sm") or 0x1F300 <= cp <= 0x1FBFF:
                seen["general"] = True
    prefix = " ".join(EMOJI_CATEGORY_TOKENS[c] for c in seen)
    return (prefix + " " + text) if prefix else text

HARMFUL_WORDS = [
    "kill","murder","hurt","attack","beat","rape","threat","die","blood","weapon",
    "stupid","idiot","loser","hate","ugly","worthless",
    "maar","dunga","dekh","chodunga","harami","gandu","chutiya",
    "madarchod","bhenchod","randi","lavde","kutte","saale",
]

def highlight_harmful_words(text: str) -> list[dict]:
    found = []
    lower = text.lower()
    for word in HARMFUL_WORDS:
        start = 0
        while True:
            idx = lower.find(word, start)
            if idx == -1: break
            found.append({
                "word" : text[idx:idx+len(word)],
                "start": idx,
                "end"  : idx + len(word),
                "type" : "violent"     if word in ["kill","murder","hurt","attack","beat","maar","dunga","die","blood","weapon"]
                         else "sexual" if word in ["rape","chodunga","randi"]
                         else "harassment"
            })
            start = idx + 1
    return found

conversation_history: list[str] = []

@torch.no_grad()
def predict(text: str, use_context: bool = False) -> dict:
    global conversation_history
    if use_context:
        conversation_history.append(text)
        if len(conversation_history) > 3:
            conversation_history.pop(0)
        full_text = " [SEP] ".join(conversation_history)
    else:
        full_text = text
    augmented = augment_with_emoji_prefix(full_text)
    enc = tokenizer(
        augmented, max_length=MAX_LENGTH,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    logits1, logits2 = model(
        input_ids      = enc["input_ids"],
        attention_mask = enc["attention_mask"],
        token_type_ids = enc.get("token_type_ids"),
    )
    probs1 = torch.softmax(logits1, dim=1)[0]
    probs2 = torch.softmax(logits2, dim=1)[0]
    pred1  = int(torch.argmax(probs1))
    pred2  = int(torch.argmax(probs2))
    level1 = L1_NAMES[pred1]
    level2 = L2_NAMES[pred2] if level1 == "threat" else None
    return {
        "level1"       : level1,
        "level2"       : level2,
        "is_threat"    : level1 == "threat",
        "confidence"   : round(float(torch.max(probs1)) * 100, 2),
        "harmful_words": highlight_harmful_words(full_text),
        "context_used" : full_text,
        "probs"        : {
            "safe"  : round(float(probs1[0]) * 100, 2),
            "threat": round(float(probs1[1]) * 100, 2),
        }
    }

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "model_loaded": model is not None})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict_api():
    load_model()
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400
    return jsonify(predict(data["text"], use_context=data.get("use_context", False)))

@app.route("/predict/conversation", methods=["POST"])
def predict_conversation():
    load_model()
    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": "Missing 'messages'"}), 400
    full_text = " [SEP] ".join(str(m) for m in data["messages"][-3:])
    return jsonify(predict(full_text))

@app.route("/clear", methods=["POST"])
def clear_context():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "context cleared"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
