"""Batch loader: memory-maps the token shards and samples random windows."""
import numpy as np
import torch

def get_batch(split, batch_size, block_size, device="cpu", data_dir="data"):
    # re-open the memmap each call so we never keep a stale/huge handle in RAM
    path = f"{data_dir}/{split}.bin"
    data = np.memmap(path, dtype=np.uint16, mode="r")

    # batch_size random start positions; leave room for a full window + the shift
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))

    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])   # 1: window of block_size ids starting at i
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])   # 2: the SAME window shifted right by one

    x, y = x.to(device), y.to(device)
    return x, y

if __name__ == "__main__":
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    x, y = get_batch("train", batch_size=4, block_size=16)
    print("x shape:", x.shape, "y shape:", y.shape)   # both (4, 16)
    print("x[0]:", tok.decode(x[0].tolist()))
    print("y[0]:", tok.decode(y[0].tolist()))