import os
import tiktoken
import zipfile
import pandas as pd
import numpy as np

data_folder = 'data/songwriter/'
# unzip zipfile to get csv dataset
with zipfile.ZipFile(os.path.join(data_folder, 'archive.zip'), 'r') as zip_f:
    zip_f.extract('spotify_millsongdata.csv', f'{data_folder}')

dataset = pd.read_csv(os.path.join(data_folder, 'spotify_millsongdata.csv'))
# concatenate all song texts
data = dataset['text'].str.cat(sep='\n')

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
