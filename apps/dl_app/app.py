"""
Emotion Classification App - BiLSTM (PyTorch)
Deployed on Hugging Face Spaces
"""

import gradio as gr
import torch
import torch.nn as nn
import re
import numpy as np
import pickle

# =========================
# LOAD VOCAB & LABELS
# =========================
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# =========================
# MODEL ARCHITECTURE
# =========================
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_class)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(self.dropout(h))

# =========================
# LOAD MODEL
# =========================
model = BiLSTM(
    vocab_size=len(vocab),
    embed_dim=128,
    hidden_dim=128,
    num_class=len(labels)
)

model.load_state_dict(torch.load("bilstm_model.pth", map_location="cpu"))
model.eval()

# =========================
# PREPROCESSING
# =========================
def clean_text(text):
    if not text or text.strip() == "":
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# =========================
# TOKENIZER (VOCAB-BASED)
# =========================
def encode(text):
    tokens = text.split()
    idx = [vocab.get(t, 0) for t in tokens]
    return torch.tensor(idx).unsqueeze(0)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_emotion(text):
    """Predict emotion from text"""
    
    if not text or text.strip() == "":
        return "Please enter some text!", None
    
    cleaned_text = clean_text(text)
    
    if cleaned_text == "":
        return "Text is empty after cleaning. Please enter meaningful text.", None
    
    x = encode(cleaned_text)
    
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1).squeeze().numpy()
    
    confidence = {
        labels[i]: float(probs[i])
        for i in range(len(labels))
    }
    
    confidence_sorted = dict(
        sorted(confidence.items(), key=lambda x: x[1], reverse=True)
    )
    
    return labels[pred], confidence_sorted

# =========================
# EXAMPLES
# =========================
examples = [
    ["I am so happy today! This is the best day ever!"],
    ["I am really angry about this situation. This is unacceptable!"],
    ["I feel so sad and lonely right now."],
    ["I am scared and worried about what will happen next."],
    ["This is absolutely disgusting and horrible."],
    ["I am so proud of what we have achieved together!"],
    ["I feel surprised by this unexpected turn of events."],
    ["I am bored and have nothing interesting to do."],
]

# =========================
# GRADIO UI (SAMA STYLE KAMU)
# =========================
with gr.Blocks(title="Emotion Classification (BiLSTM)", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🎭 Emotion Classification App (BiLSTM)
        ### Classify emotions from text using Deep Learning (PyTorch)
        
        This app uses a **BiLSTM model** trained on 20 different emotions.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Enter your text",
                placeholder="Type something here... (e.g., 'I am so happy today!')",
                lines=5
            )
            
            predict_btn = gr.Button("🔮 Predict Emotion", variant="primary")
            clear_btn = gr.ClearButton([text_input])
            
            gr.Examples(
                examples=examples,
                inputs=text_input,
                label="Try these examples:"
            )
        
        with gr.Column(scale=1):
            emotion_output = gr.Textbox(
                label="Predicted Emotion",
                interactive=False
            )
            
            confidence_output = gr.Label(
                label="Confidence Scores (Top Emotions)"
            )
    
    gr.Markdown(
        """
        ---
        ### 📊 About Model
        
        - Model: BiLSTM (PyTorch)
        - Features: Word Embedding (learned)
        - Classes: 20 emotions
        - Deployment: Hugging Face Spaces
        
        ---
        """
    )
    
    predict_btn.click(
        fn=predict_emotion,
        inputs=text_input,
        outputs=[emotion_output, confidence_output]
    )

# =========================
# LAUNCH
# =========================
if __name__ == "__main__":
    demo.launch()