import torch
from torch.utils.data import Dataset
import json
import torch
import torch.nn as nn
import sentencepiece as spm
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from RNNLanguageModel import RNNLanguageModel

"""Step 4"""

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
NUM_LAYERS = 2
LEARNING_RATE = 1e-4
EPOCHS = 30
PATIENCE = 3
MODEL_PATH = "rnn.pt"

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    vocab_size = tokenizer.get_piece_size()

    train_dataset = TextDataset(TRAIN_FILE, tokenizer, MAX_SEQ_LEN)
    val_dataset = TextDataset(VAL_FILE, tokenizer, MAX_SEQ_LEN)  

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = RNNLanguageModel(vocab_size=vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=3) 

    best_val_loss = float('inf')
    no_improvement_epochs = 0

    train_losses, val_losses = [], []
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        for input_ids, target_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            input_ids  = input_ids.to(device) 
            target_ids = target_ids.to(device)
            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                logits, _ = model(input_ids)
                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch : {epoch+1}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), MODEL_PATH)
            
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= PATIENCE:
                print("Stopping Early.")
                break
    
    return train_losses,val_losses

train_losses,val_losses = train_model()

import matplotlib.pyplot as plt
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss-rnn.png')
