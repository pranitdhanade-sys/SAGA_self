import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Download nltk data (only first time)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Dataset (LOCAL ONLY)
# -----------------------------

def load_data(file_path):
    texts = []
    labels = []

    with open(file_path, "r") as f:
        for line in f:
            text, label = line.strip().split(";")
            texts.append(text)
            labels.append(label)

    return texts, labels


train_texts, train_labels = load_data("train.txt")
val_texts, val_labels = load_data("val.txt")
test_texts, test_labels = load_data("test.txt")

# Encode labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)
test_labels = label_encoder.transform(test_labels)

num_classes = len(label_encoder.classes_)

# -----------------------------
# Text Preprocessing
# -----------------------------

stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return tokens

train_texts = [preprocess(t) for t in train_texts]
val_texts = [preprocess(t) for t in val_texts]
test_texts = [preprocess(t) for t in test_texts]

# Build vocabulary
vocab = {"<PAD>": 0, "<UNK>": 1}
for sentence in train_texts:
    for word in sentence:
        if word not in vocab:
            vocab[word] = len(vocab)

def encode(sentence, max_len=20):
    encoded = [vocab.get(word, vocab["<UNK>"]) for word in sentence]
    if len(encoded) < max_len:
        encoded += [vocab["<PAD>"]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return encoded

train_encoded = [encode(s) for s in train_texts]
val_encoded = [encode(s) for s in val_texts]
test_encoded = [encode(s) for s in test_texts]

# -----------------------------
# Dataset Class
# -----------------------------

class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


train_dataset = EmotionDataset(train_encoded, train_labels)
val_dataset = EmotionDataset(val_encoded, val_labels)
test_dataset = EmotionDataset(test_encoded, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# -----------------------------
# Model
# -----------------------------

class EmotionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out


model = EmotionModel(len(vocab)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training Loop
# -----------------------------

def train():
    model.train()
    total_loss = 0

    for texts, labels in tqdm(train_loader):
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# -----------------------------
# Run Training
# -----------------------------

EPOCHS = 5

for epoch in range(EPOCHS):
    loss = train()
    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {loss:.4f} | Val Accuracy: {val_acc:.4f}")

test_acc = evaluate(test_loader)
print(f"\nTest Accuracy: {test_acc:.4f}")



# -----------------------------
# Save Model + Artifacts
# -----------------------------

SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "emotion_model.pt")
VOCAB_PATH = os.path.join(SAVE_DIR, "vocab.npy")
LABEL_PATH = os.path.join(SAVE_DIR, "label_classes.npy")

# Save model weights
torch.save(model.state_dict(), MODEL_PATH)

# Save vocabulary
np.save(VOCAB_PATH, vocab)

# Save label encoder classes
np.save(LABEL_PATH, label_encoder.classes_)

print("\n✅ Model Saved Successfully")
print(f"Model -> {MODEL_PATH}")
print(f"Vocab -> {VOCAB_PATH}")
print(f"Labels -> {LABEL_PATH}")


# -----------------------------
# Save Model
# -----------------------------

torch.save({
    "model_state_dict": model.state_dict(),
    "vocab": vocab,
    "classes": label_encoder.classes_.tolist()
}, "emotion_model.pth")

print("Model saved safely as emotion_model.pth")

