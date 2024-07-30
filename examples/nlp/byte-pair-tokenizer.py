import matplotlib.pyplot as plt

with open("data/random.txt", "r") as f:
    text = f.read()

bytes_input = list(text.encode("UTF-8"))
token_seq = bytes_input.copy()

token_seq_lens = [len(token_seq)]
vocabulary_lens = [len(set(token_seq))]

n_loops = 30
next_token_code = 256
merges = {}

for x in range(n_loops):
    counts = {}

    for pair in zip(token_seq, token_seq[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    pair_max = max(counts, key=counts.get)

    pair_pos = []
    for i, pair in enumerate(zip(token_seq, token_seq[1:])):
        if pair == pair_max:
            pair_pos.append(i)

    print(f"found {len(pair_pos)} pair occurences of {pair_max}")

    merges[next_token_code] = pair_max
        
    for pos in reversed(pair_pos):
        token_seq[pos] = next_token_code
        del token_seq[pos+1]

    next_token_code += 1

    token_seq_lens.append(len(token_seq))
    vocabulary_lens.append(len(set(token_seq)))

merges_inv = {v: k for k, v in merges.items()}

plt.figure(figsize=(10, 5))
plt.plot(vocabulary_lens, token_seq_lens, label='Input Lengths')
plt.xlabel('Vocabulary Size')
plt.ylabel('Token Sequence Length')
plt.show()

print("Compression ratio:", token_seq_lens[0] / token_seq_lens[-1])

def _decode(x):
    if x in merges:
        x1 = _decode(merges[x][0])
        x2 = _decode(merges[x][1])
        return x1 + x2
    else:
        return [x]

def decode(ids):
    out = []
    for idx in ids:
        sub = _decode(idx)
        out.extend(sub)
    
    return bytes(out).decode()

def encode(text):
    btstream = list(text.encode('UTF-8'))

    while True:
        found = False
        ids_to_delete = []
        for i, b in enumerate(zip(btstream, btstream[1:])):
            if b in merges_inv:
                found = True
                btstream[i] = merges_inv[b]
                ids_to_delete.append(i+1)

        for idx in reversed(ids_to_delete):
            del btstream[idx]

        if not found:
            break

    return btstream


input_txt = """Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes"""

encoded = encode(input_txt)
decoded = decode(encoded)

assert input_txt == decoded