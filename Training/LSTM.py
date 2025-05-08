# Install dependencies
!pip install nltk evaluate --quiet

# === Imports ===
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import nltk
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate

nltk.download('punkt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load and preprocess data ===
df = pd.read_csv("/content/train.csv").dropna(subset=['article', 'highlights']).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)
df['article'] = df['article'].apply(lambda x: word_tokenize(x.lower()))
df['highlights'] = df['highlights'].apply(lambda x: word_tokenize(x.lower()))

# === Build vocabulary ===
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"

def build_vocab(token_lists, min_freq=2):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

src_vocab = build_vocab(df['article'])
tgt_vocab = build_vocab(df['highlights'])
src_idx2word = {i: w for w, i in src_vocab.items()}
tgt_idx2word = {i: w for w, i in tgt_vocab.items()}

# === Dataset class ===
class SummarizationDataset(Dataset):
    def __init__(self, articles, summaries, src_vocab, tgt_vocab, max_len=100):
        self.articles = articles
        self.summaries = summaries
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.articles)

    def encode(self, tokens, vocab):
        return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens[:self.max_len]]

    def __getitem__(self, idx):
        src = self.encode(self.articles[idx], self.src_vocab)
        tgt = [self.tgt_vocab[SOS_TOKEN]] + self.encode(self.summaries[idx], self.tgt_vocab) + [self.tgt_vocab[EOS_TOKEN]]
        return torch.tensor(src), torch.tensor(tgt)

# === Model: Encoder, Decoder, Attention ===
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim*2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        B, src_len, _ = encoder_outputs.shape
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM((enc_hid_dim*2) + emb_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear((enc_hid_dim*2) + dec_hid_dim + emb_dim, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        output = torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1)
        return self.fc_out(output), hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        B, tgt_len = tgt.shape
        outputs = torch.zeros(B, tgt_len, self.decoder.output_dim).to(device)
        encoder_outputs, hidden, cell = self.encoder(src)
        hidden = hidden[-1].unsqueeze(0)
        cell = cell[-1].unsqueeze(0)
        input = tgt[:, 0]
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        return outputs

# === Summarize Function ===
def summarize(text, model, max_len=50):
    model.eval()
    tokens = word_tokenize(text.lower())
    input_ids = [src_vocab.get(tok, src_vocab[UNK_TOKEN]) for tok in tokens]
    src_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        hidden = hidden[-1].unsqueeze(0)
        cell = cell[-1].unsqueeze(0)
        input_token = torch.tensor([tgt_vocab[SOS_TOKEN]]).to(device)
        summary_ids = []

        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            next_token = output.argmax(1).item()
            if next_token == tgt_vocab[EOS_TOKEN]:
                break
            summary_ids.append(next_token)
            input_token = torch.tensor([next_token]).to(device)

    return ' '.join([tgt_idx2word[i] for i in summary_ids])

# === Evaluation Function ===
rouge = evaluate.load("rouge")

def evaluate_model(model, val_df, num_samples=100):
    bleu_scores = []
    predictions, references = [], []

    for i in range(min(num_samples, len(val_df))):
        src_text = ' '.join(val_df.iloc[i]['article'])
        ref_text = ' '.join(val_df.iloc[i]['highlights'])
        pred_text = summarize(src_text, model)
        predictions.append(pred_text)
        references.append(ref_text)

        bleu = sentence_bleu([ref_text.split()], pred_text.split(), smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(bleu)

    print(f"\nBLEU: {np.mean(bleu_scores):.4f}")
    print("ROUGE:", rouge.compute(predictions=predictions, references=references))

# === DataLoaders ===
train_data, val_data = train_test_split(df, test_size=0.1)
train_ds = SummarizationDataset(train_data['article'], train_data['highlights'], src_vocab, tgt_vocab)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=lambda batch: [
    torch.nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True),
    torch.nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True)], drop_last=True)

# === Model Setup ===
INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(tgt_vocab)
EMB_DIM = 256
HID_DIM = 512
attn = Attention(HID_DIM, HID_DIM)
enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, HID_DIM, attn)
model = Seq2Seq(enc, dec).to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)

# === Train Loop ===
for epoch in range(5):
    model.train()
    total_loss = 0
    for src_batch, tgt_batch in train_dl:
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        outputs = model(src_batch, tgt_batch)
        loss = criterion(outputs[:, 1:].reshape(-1, OUTPUT_DIM), tgt_batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_dl):.4f}")

# === Evaluate + Inference ===
evaluate_model(model, val_data.reset_index(drop=True), num_samples=100)

sample = "The Mars rover Perseverance has been exploring the Red Planet collecting rock samples."
print("Summary:", summarize(sample, model))
