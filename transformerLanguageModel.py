import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class transformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=6, ff_dim=512, dropout=0.2, pad_token_id=0):
        super(transformerLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        encoder_layers = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)

        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask    
    
    def forward(self, input_ids):
        mask = self._generate_square_subsequent_mask(input_ids.size(1))
        mask = mask.to(input_ids.device)
        out = self.embedding(input_ids)*math.sqrt(self.embedding.embedding_dim)
        out = self.positional_encoding(out)
        out = self.transformer_encoder(out, mask)
        out = self.fc(out)

        return out
    
    def predict_next_token(self, input_ids, temperature=1.0):
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            #next_token_id = torch.argmax(probs, dim=-1)
            
            if temperature <= 0.01:  
                next_token_id = torch.argmax(probs, dim=-1)
            else:  
                next_token_id = torch.multinomial(probs, 1).squeeze()
            
            
            return next_token_id.item() 
    
    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, temperature=1.0, device = "cuda"):
        self.eval()
        input_ids = tokenizer.encode(prompt, out_type=int)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        generated_ids = []
        hidden = None

        for _ in range(max_length):
            next_token_id = self.predict_next_token(input_tensor, temperature)
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            generated_ids.append(next_token_id)
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        return tokenizer.decode(generated_ids, out_type=str)

