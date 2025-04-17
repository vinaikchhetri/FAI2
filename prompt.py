import torch
from LSTMLanguageModel import LSTMLanguageModel
import sentencepiece as spm
from GRULanguageModel import GRULanguageModel
from RNNLanguageModel import RNNLanguageModel
from transformerLanguageModel import transformerLanguageModel



def test_models_with_prompt(model_path, tokenizer, prompt, max_length=50, temperature=0.01, device="cpu"):

    model = LSTMLanguageModel(vocab_size=tokenizer.get_piece_size(), embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.2, pad_token_id=0)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
 
    response = model.generate(
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        eos_token_id=2, 
        temperature=temperature,
        device=device
    )

    
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

TOKENIZER_PATH = "bpe_tokenizer.model"
tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
test_models_with_prompt("lstm.pt", tokenizer, "Which do you prefer? Dogs or cats?", max_length=50, temperature=1, device="cpu")

test_models_with_prompt("lstm.pt", tokenizer, "I like to have chips because ", max_length=50, temperature=1, device="cpu")

print("\n\n")



def gru_test_models_with_prompt(model_path, tokenizer, prompt, max_length=50, temperature=0.01, device="cpu"):
   

    model = GRULanguageModel(vocab_size=tokenizer.get_piece_size(), embed_dim=256, hidden_dim=512, num_layers=6,dropout=0.2, pad_token_id=0 )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


    response = model.generate(
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        eos_token_id=2,
        temperature=temperature,
        device=device
    )

    print(f"Model: {model_path}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

TOKENIZER_PATH = "bpe_tokenizer.model"
tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
gru_test_models_with_prompt("gru.pt", tokenizer, "Which do you prefer? Dogs or cats?", max_length=50, temperature=1, device="cpu")

gru_test_models_with_prompt("gru.pt", tokenizer, "I like to have chips because ", max_length=50, temperature=1, device="cpu")

print("\n\n")

def rnn_test_models_with_prompt(model_path, tokenizer, prompt, max_length=50, temperature=0.01, device="cpu"):
    model = RNNLanguageModel(vocab_size=tokenizer.get_piece_size(), embed_dim=256, hidden_dim=512, num_layers=2,dropout=0.2, pad_token_id=0)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


    response = model.generate(
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        eos_token_id=2,
        temperature=temperature,
        device=device
    )

    print(f"Model: {model_path}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

TOKENIZER_PATH = "bpe_tokenizer.model"
tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
rnn_test_models_with_prompt("rnn.pt", tokenizer, "Which do you prefer? Dogs or cats?", max_length=50, temperature=1, device="cpu")
rnn_test_models_with_prompt("rnn.pt", tokenizer, "I like to have chips because ", max_length=50, temperature=1, device="cpu")

print("\n\n")

def transformer_test_models_with_prompt(model_path, tokenizer, prompt, max_length=50, temperature=0.01, device="cpu"):
    
    
    model = transformerLanguageModel(vocab_size=tokenizer.get_piece_size(), embed_dim=256, num_heads = 8, num_layers=6, ff_dim=512, dropout=0.2, pad_token_id=0)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


    response = model.generate(
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        eos_token_id=2,
        temperature=temperature,
        device=device
    )

    print(f"Model: {model_path}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

TOKENIZER_PATH = "bpe_tokenizer.model"
tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
transformer_test_models_with_prompt("transformer.pt", tokenizer, "Which do you prefer? Dogs or cats?", max_length=50, temperature=1, device="cpu")
transformer_test_models_with_prompt("transformer.pt", tokenizer, "I like to have chips because ", max_length=50, temperature=1, device="cpu")
print("\n\n")
