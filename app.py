"""
app.py  —  Cyber Threat Detection API
Model loads lazily on first request — server starts immediately without crashing.
"""

from __future__ import annotations
import os, unicodedata, threading
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://cybershield-frontend-jqz8vl3cp-ritupuranik2-dots-projects.vercel.app",
                   "https://*.vercel.app"])

DEVICE     = torch.device("cpu")
MODEL_DIR  = Path("saved_model")
HF_REPO    = os.environ.get("HF_REPO", "").strip()
HF_TOKEN   = os.environ.get("HF_TOKEN", "").strip() or None
MAX_LENGTH = 128

L1_NAMES = {0: "safe",       1: "threat"}
L2_NAMES = {0: "harassment", 1: "violent", 2: "rape"}

_model      = None
_tokenizer  = None
_load_error = None


class HierarchicalThreatClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.3):
        super().__init__()
        from transformers import AutoConfig, AutoModel
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


def _load_model():
    global _model, _tokenizer, _load_error
    if _model is not None:
        return True
    try:
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download

        if not MODEL_DIR.exists():
            if not HF_REPO:
                raise RuntimeError("HF_REPO env var not set")
            print(f"Downloading model: {HF_REPO}")
            snapshot_download(repo_id=HF_REPO, local_dir=str(MODEL_DIR), token=HF_TOKEN)
            print("Download complete!")

        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model     = HierarchicalThreatClassifier(str(MODEL_DIR))
        heads      = torch.load(MODEL_DIR / "heads.pt", map_location=DEVICE)
        _model.head1.load_state_dict(heads["head1"])
        _model.head2.load_state_dict(heads["head2"])
        _model.to(DEVICE).eval()
        _load_error = None
        print("Model ready!")
        return True
    except Exception as e:
        _load_error = str(e)
        print(f"Model load error: {e}")
        return False


AGGRESSIVE_EMOJIS = set("😡🤬👊🔪💀☠️🔫💣⚔️🖕🩸")
SEXUAL_EMOJIS     = set("🍆🍑😈💦🔞")
EMOJI_TOKENS      = {"aggressive": "[EMOJI_AGGR]", "sexual": "[EMOJI_SEX]", "general": "[EMOJI_GEN]"}

def augment_with_emoji_prefix(text: str) -> str:
    seen: dict[str, bool] = {}
    for ch in text:
        if ch in AGGRESSIVE_EMOJIS:   seen["aggressive"] = True
        elif ch in SEXUAL_EMOJIS:     seen["sexual"]     = True
        else:
            cp = ord(ch); cat = unicodedata.category(ch)
            if cat in ("So","Sm") or 0x1F300 <= cp <= 0x1FBFF:
                seen["general"] = True
    prefix = " ".join(EMOJI_TOKENS[c] for c in seen)
    return (prefix + " " + text) if prefix else text


HARMFUL_WORDS = [
    "kill","murder","hurt","attack","beat","rape","threat","die","blood",
    "stupid","idiot","loser","hate","ugly","worthless",
    "maar","dunga","harami","gandu","chutiya","madarchod","bhenchod","randi",
]

def highlight_harmful_words(text: str) -> list[dict]:
    found = []; lower = text.lower()
    for word in HARMFUL_WORDS:
        start = 0
        while True:
            idx = lower.find(word, start)
            if idx == -1: break
            found.append({
                "word": text[idx:idx+len(word)], "start": idx, "end": idx+len(word),
                "type": "violent" if word in ["kill","murder","hurt","attack","beat","maar","dunga","die","blood"]
                        else "sexual" if word in ["rape","randi"] else "harassment"
            })
            start = idx + 1
    return found


conversation_history: list[str] = []

@torch.no_grad()
def predict(text: str, use_context: bool = False) -> dict:
    global conversation_history
    if use_context:
        conversation_history.append(text)
        if len(conversation_history) > 3: conversation_history.pop(0)
        full_text = " [SEP] ".join(conversation_history)
    else:
        full_text = text

    augmented = augment_with_emoji_prefix(full_text)
    enc = _tokenizer(augmented, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    logits1, logits2 = _model(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        token_type_ids=enc.get("token_type_ids"),
    )
    probs1 = torch.softmax(logits1, dim=1)[0]
    probs2 = torch.softmax(logits2, dim=1)[0]
    pred1  = int(torch.argmax(probs1))
    level1 = L1_NAMES[pred1]
    level2 = L2_NAMES[int(torch.argmax(probs2))] if level1 == "threat" else None

    return {
        "level1": level1, "level2": level2,
        "is_threat": level1 == "threat",
        "confidence": round(float(torch.max(probs1)) * 100, 2),
        "harmful_words": highlight_harmful_words(full_text),
        "context_used": full_text,
        "probs": {"safe": round(float(probs1[0])*100,2), "threat": round(float(probs1[1])*100,2)}
    }


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "model_loaded": _model is not None})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": _model is not None, "load_error": _load_error})

@app.route("/predict", methods=["POST"])
def predict_api():
    if not _load_model():
        return jsonify({"error": f"Model not loaded yet: {_load_error}"}), 503
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text'"}), 400
    return jsonify(predict(data["text"], use_context=data.get("use_context", False)))

@app.route("/predict/conversation", methods=["POST"])
def predict_conversation():
    if not _load_model():
        return jsonify({"error": f"Model not loaded yet: {_load_error}"}), 503
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


# Load model in background — server starts instantly
threading.Thread(target=_load_model, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
