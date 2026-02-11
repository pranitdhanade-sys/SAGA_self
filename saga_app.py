import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from emotion_pipeline import preprocess, encode, EmotionModel, vocab, DEVICE
from comext import get_comments

# -----------------------------
# Universal Model Loader
# -----------------------------
def load_model_safely(model, path):
    import os
    ext = os.path.splitext(path)[1].lower()
    label_encoder = None

    if ext in [".pth", ".pt"]:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, torch.nn.Module):
            return checkpoint.to(DEVICE), None
        elif isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif all(k.startswith(("weight", "bias")) or "." in k for k in checkpoint.keys()):
                model.load_state_dict(checkpoint)
            if "label_encoder" in checkpoint:
                label_encoder = checkpoint["label_encoder"]
        return model.to(DEVICE), label_encoder

    elif ext == ".npy":
        weights = torch.tensor(np.load(path, allow_pickle=True))
        try:
            model.load_state_dict(weights)
        except:
            print("[WARNING] Could not load .npy weights directly")
        return model.to(DEVICE), None
    else:
        raise ValueError(f"Unsupported model format: {ext}")

# -----------------------------
# Load model
# -----------------------------
model = EmotionModel(len(vocab)).to(DEVICE)
model, label_encoder = load_model_safely(model, "emotion_model.pth")
model.eval()

# -----------------------------
# Predict emotion
# -----------------------------
def predict_emotion(text):
    tokens = preprocess(text)
    encoded = encode(tokens)
    tensor = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return probs

# -----------------------------
# Plot emotions
# -----------------------------
def plot_emotions(probs):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(range(len(probs)), probs)
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels([f"E{i}" for i in range(len(probs))])
    ax.set_ylim(0,1)
    ax.set_ylabel("Probability")
    ax.set_title("Emotion Probabilities")
    return fig

# -----------------------------
# Tkinter GUI
# -----------------------------
class SAGAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SAGA YouTube Emotion Analyzer")

        ttk.Label(root, text="YouTube URL:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.url_entry = ttk.Entry(root, width=60)
        self.url_entry.grid(row=0, column=1, padx=5, pady=5)

        self.analyze_button = ttk.Button(root, text="Analyze", command=self.analyze)
        self.analyze_button.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(root, text="Comments:").grid(row=1, column=0, sticky="nw", padx=5)
        self.comment_box = ScrolledText(root, width=80, height=15)
        self.comment_box.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

        ttk.Label(root, text="Emotion Graph:").grid(row=2, column=0, sticky="nw", padx=5)
        self.canvas_frame = ttk.Frame(root)
        self.canvas_frame.grid(row=2, column=1, columnspan=2, padx=5, pady=5)
        self.canvas = None

    def analyze(self):
        url = self.url_entry.get().strip()
        if "v=" not in url:
            messagebox.showerror("Error", "Invalid YouTube URL")
            return
        video_id = url.split("v=")[-1]

        self.comment_box.delete("1.0", tk.END)
        try:
            comments = get_comments(video_id, max_comments=50)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch comments:\n{e}")
            return

        for c in comments:
            self.comment_box.insert(tk.END, c["comment"] + "\n\n")

        # Average probabilities
        avg_probs = None
        for c in comments:
            probs = predict_emotion(c["comment"])
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs += probs
        avg_probs /= len(comments)

        # Plot
        fig = plot_emotions(avg_probs)
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

# -----------------------------
# Run Tkinter app
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SAGAApp(root)
    root.mainloop()

