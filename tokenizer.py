from typing import Tuple
import json
import os
import sentencepiece as spm

"""Step1"""
def add_special_tokens(pairs: Tuple[list]):
    new_prompts = []
    new_completions = []

    for prompt, completion in zip(pairs):
        if prompt[0].isupper():
            prompt = '<bos>' + prompt

        if completion.endswith('.') or complettion.endswith('?') or completion.endswith('!'):
            completion += '<eos>'

        new_prompts.append(prompt)
        new_completions.append(completion)

    return new_prompts, new_completions


"""Step 2"""
def merge_text_files(data_dir, output_file):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                    outfile.write(f.read())


if __name__ == "__main__":
    
    """Step 2"""

    DATA_DIR = "data/raw"
    TOKENIZER_PREFIX = "bpe_tokenizer"
    VOCAB_SIZE = 10000
    CORPUS_FILE = "corpus.txt"
    merge_text_files(DATA_DIR, CORPUS_FILE)

    spm.SentencePieceTrainer.train(
        input=CORPUS_FILE,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=VOCAB_SIZE,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=",".join(["<bos>", "<eos>", "<pad>"])
    )
    print("Tokenizer training complete! Files generated:")
    print(f"{TOKENIZER_PREFIX}.model")
    print(f"{TOKENIZER_PREFIX}.vocab")
