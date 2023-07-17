"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()


def encode(s):
    return [stoi[c] if c in chars else 0 for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    return ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# get all unique characters from the smallest dataset slice
chars = sorted(list(set(data[:int(len(data) * 0.1)])))

for slice_size in np.arange(0.9, 0, -0.1):
    temp_data = data[:int(len(data) * slice_size)]
    temp_dataset_len = len(temp_data)
    print(f"length of dataset in characters: {temp_dataset_len}")

    # get all the unique characters that occur in this text
    # chars = sorted(list(set(temp_data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # create the train and test splits
    train_data = temp_data[:int(temp_dataset_len * 0.9)]
    val_data = temp_data[int(temp_dataset_len * 0.9):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), f'train{int(slice_size*100)}.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), f'val{int(slice_size*100)}.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }

    with open(os.path.join(os.path.dirname(__file__), f'meta{int(slice_size*100)}.pkl'), 'wb') as f:
        pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
