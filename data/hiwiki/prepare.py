"""
Saves the WikiPedia dataset to a binary file for training.

From https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
"""

import os
import glob

import numpy as np
from datasets import load_dataset
# from tokenizers import Tokenizer
import tiktoken
from tqdm import tqdm

NUM_PROCESSES = 8
TOTAL_BATCHES = 1024

data_dir = "./textmd"
data_paths = glob.glob(os.path.join(data_dir, "**"), recursive=True)
data_files = [item_path for item_path in data_paths if os.path.isfile(item_path)]

# tokenizer = Tokenizer.from_file("../hi_bpe_tokenizer.json")
tokenizer = tiktoken.get_encoding("gpt2")

# about 200K documents (221,340)
dataset = load_dataset("text", data_files=data_files, num_proc=NUM_PROCESSES)

split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True) # type: ignore[attr-defined]
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

def process(example):
    # ids = tokenizer.encode("[START]") + tokenizer.encode(example['text']) + tokenizer.encode("[END]")
    # out = {'ids': ids.ids, 'len': len(ids.ids)}
    ids = tokenizer.encode_ordinary(example['text']) # ignores any special tokens
    ids.append(tokenizer.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=NUM_PROCESSES
)

# concatenate the ids in each dataset into one large file to use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))

    idx = 0
    for batch_idx in tqdm(range(TOTAL_BATCHES), desc=f"writing {split}.bin", total=TOTAL_BATCHES):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=TOTAL_BATCHES, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

# train.bin is ~104MB, val.bin ~60KB
# train has ~54M tokens (54,502,782)
# val has ~30K tokens (30,235)

# to read the bin files
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')
