import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

import sentencepiece as spm
from torch.utils.data import DataLoader
from GRULanguageModel import GRULanguageModel
from torch.utils.data import Dataset
import json

from LSTMLanguageModel import LSTMLanguageModel
from RNNLanguageModel import RNNLanguageModel
from transformerLanguageModel import transformerLanguageModel


class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_seq_len=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item["prompt"] + " " + item["completion"]
                token_ids = tokenizer.encode(text, out_type=int)[:max_seq_len]
                if len(token_ids) < 2:
                    continue
                self.samples.append(token_ids)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        token_ids = self.samples[idx]
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        return input_ids, target_ids
    

def collate_fn(batch):
    input_batch, target_batch = zip(*batch)
    input_batch = nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=3)
    target_batch = nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=3)
    return input_batch, target_batch





TOKENIZER_PATH = "bpe_tokenizer.model"
TRAIN_FILE = "data/train.jsonl"
VAL_FILE = "data/test.jsonl"
MAX_SEQ_LEN = 512
BATCH_SIZE = 128
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 6
LEARNING_RATE = 1e-4
EPOCHS = 30
PATIENCE = 3
MODEL_PATH = "gru.pt"

tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GRULanguageModel(vocab_size=tokenizer.get_piece_size(), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


test_dataset = TextDataset(VAL_FILE, tokenizer, MAX_SEQ_LEN)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

total_loss = 0.0
total_tokens = 0
criterion = nn.CrossEntropyLoss(ignore_index=3, reduction='sum')

reference_list = []
hypothesis_list = []

with torch.no_grad():
    for input_ids, target_ids in test_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits, _ = model(input_ids)
        loss = criterion(logits.view(-1, tokenizer.get_piece_size()), target_ids.view(-1))
        total_loss += loss.item()
        total_tokens += (target_ids != 3).sum().item()

        inp = tokenizer.decode(input_ids[0].tolist(), out_type=str)
        out = model.generate(tokenizer, inp, max_length=MAX_SEQ_LEN, eos_token_id=2,temperature=1, device=device)
        reference = tokenizer.decode(target_ids[0].tolist(), out_type=str)

        reference_list.append([reference.split()])
        hypothesis_list.append(out.split())


perplexity = math.exp(total_loss / total_tokens)
smoother = SmoothingFunction().method1
bleu_score = corpus_bleu(reference_list, hypothesis_list, smoothing_function=smoother)

print("GRU")
print(f"Perplexity: {perplexity:.4f}")
print(f"BLEU Score: {bleu_score:.4f}")



TOKENIZER_PATH = "bpe_tokenizer.model"
TRAIN_FILE = "data/train.jsonl"
VAL_FILE = "data/test.jsonl"
MAX_SEQ_LEN = 512
BATCH_SIZE = 128
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
LEARNING_RATE = 1e-4
EPOCHS = 30
PATIENCE = 3
MODEL_PATH = "lstm.pt"

tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMLanguageModel(vocab_size=tokenizer.get_piece_size(), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()



test_dataset = TextDataset(VAL_FILE, tokenizer, MAX_SEQ_LEN)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

total_loss = 0.0
total_tokens = 0
criterion = nn.CrossEntropyLoss(ignore_index=3, reduction='sum')  

reference_list = []
hypothesis_list = []

with torch.no_grad():
    for input_ids, target_ids in test_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits, _ = model(input_ids)
        loss = criterion(logits.view(-1, tokenizer.get_piece_size()), target_ids.view(-1))
        total_loss += loss.item() 
        total_tokens += (target_ids != 3).sum().item()

        inp = tokenizer.decode(input_ids[0].tolist(), out_type=str)
        out = model.generate(tokenizer, inp, max_length=MAX_SEQ_LEN, eos_token_id=2, temperature=1, device=device)
        reference = tokenizer.decode(target_ids[0].tolist(), out_type=str)

        reference_list.append([reference.split()])
        hypothesis_list.append(out.split())


perplexity = math.exp(total_loss / total_tokens)
smoother = SmoothingFunction().method1
bleu_score = corpus_bleu(reference_list, hypothesis_list, smoothing_function=smoother)

print("LSTM")
print(f"Perplexity: {perplexity:.4f}")
print(f"BLEU Score: {bleu_score:.4f}")




TOKENIZER_PATH = "bpe_tokenizer.model"
TRAIN_FILE = "data/train.jsonl"
VAL_FILE = "data/test.jsonl"
MAX_SEQ_LEN = 512
BATCH_SIZE = 128
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
LEARNING_RATE = 1e-4
EPOCHS = 30
PATIENCE = 3
MODEL_PATH = "rnn.pt"

tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RNNLanguageModel(vocab_size=tokenizer.get_piece_size(), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

test_dataset = TextDataset(VAL_FILE, tokenizer, MAX_SEQ_LEN)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

total_loss = 0.0
total_tokens = 0
criterion = nn.CrossEntropyLoss(ignore_index=3, reduction='sum')  

reference_list = []
hypothesis_list = []

with torch.no_grad():
    for input_ids, target_ids in test_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits, _ = model(input_ids)
        loss = criterion(logits.view(-1, tokenizer.get_piece_size()), target_ids.view(-1))
        total_loss += loss.item() 
        total_tokens += (target_ids != 3).sum().item()

        inp = tokenizer.decode(input_ids[0].tolist(), out_type=str)
        out = model.generate(tokenizer, inp, max_length=MAX_SEQ_LEN, eos_token_id=2, temperature=1, device=device)
        reference = tokenizer.decode(target_ids[0].tolist(), out_type=str)

        reference_list.append([reference.split()])
        hypothesis_list.append(out.split())


perplexity = math.exp(total_loss / total_tokens)
smoother = SmoothingFunction().method1
bleu_score = corpus_bleu(reference_list, hypothesis_list, smoothing_function=smoother)
print("rnn")
print(f"Perplexity: {perplexity:.4f}")
print(f"BLEU Score: {bleu_score:.4f}")




TOKENIZER_PATH = "bpe_tokenizer.model"
TRAIN_FILE = "data/train.jsonl"
VAL_FILE = "data/test.jsonl"
MAX_SEQ_LEN = 512
BATCH_SIZE = 128
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 6
LEARNING_RATE = 1e-4
EPOCHS = 30
PATIENCE = 3
MODEL_PATH = "transformer.pt"
NUM_HEADS = 8

tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = transformerLanguageModel(vocab_size=tokenizer.get_piece_size(), embed_dim=EMBED_DIM, num_heads=8, num_layers=NUM_LAYERS, ff_dim=512, dropout=0.2, pad_token_id=0)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


test_dataset = TextDataset(VAL_FILE, tokenizer, MAX_SEQ_LEN)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

total_loss = 0.0
total_tokens = 0
criterion = nn.CrossEntropyLoss(ignore_index=3, reduction='sum')  

reference_list = []
hypothesis_list = []

with torch.no_grad():
    for input_ids, target_ids in test_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)
        loss = criterion(logits.view(-1, tokenizer.get_piece_size()), target_ids.view(-1))
        total_loss += loss.item() 
        total_tokens += (target_ids != 3).sum().item()

        inp = tokenizer.decode(input_ids[0].tolist(), out_type=str)
        
        out = model.generate(tokenizer, inp, max_length=MAX_SEQ_LEN, eos_token_id=2,temperature=1,  device=device)
        reference = tokenizer.decode(target_ids[0].tolist(), out_type=str)

        reference_list.append([reference.split()])
        hypothesis_list.append(out.split())


perplexity = math.exp(total_loss / total_tokens)
smoother = SmoothingFunction().method1
bleu_score = corpus_bleu(reference_list, hypothesis_list, smoothing_function=smoother)

print("tranformer")
print(f"Perplexity: {perplexity:.4f}")
print(f"BLEU Score: {bleu_score:.4f}")
