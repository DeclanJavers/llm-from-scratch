# the start of my tokenizer

text = "the cat sat on the mat"
tokens = list(text.encode("utf-8"))
print(tokens)
print(len(tokens))

# gets the most common two tokens next to each other
def get_stats(tokens): 
    counts = {}
    for i in range(len(tokens) - 1):
        current_token = tokens[i]
        next_token = tokens[i + 1]
        key = (current_token, next_token)
        counts[key] = counts.get(key, 0) + 1
    return counts

def merge(tokens, pair, new_id):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(new_id)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def train(tokens, vocab_size):
    num_merges = vocab_size - 256          # 256 bytes already exist
    tokens = list(tokens)                  # copy so we don't clobber the original
    merges = {}
    for step in range(num_merges):
        stats = get_stats(tokens)
        best_pair = max(stats, key=stats.get)
        new_id = 256 + step
        tokens = merge(tokens, best_pair, new_id)
        merges[best_pair] = new_id
        print(f"merge {step}: {best_pair} -> {new_id}")
    return tokens, merges

final_tokens, merges = train(tokens, vocab_size=260)   # just 4 merges to start
print("final length:", len(final_tokens))

def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for pair, new_id in merges.items():
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]  # glue the two pieces' bytes together
    tokens_bytes = b"".join(vocab[id] for id in ids)     # look up each id's bytes
    return tokens_bytes.decode("utf-8")

print(decode([259, 99, 97, 116], merges))