from typing import List, Tuple, Dict

def build_vocab() -> Tuple[list, Dict[str, int]]:
    specials = ['<pad>', '<bos>', '<eos>']
    lowers = [chr(ord('a') + i) for i in range(26)]
    uppers = [chr(ord('A') + i) for i in range(26)]
    digits = [str(i) for i in range(10)]
    symbols = list(" .,:;!?-_/+*=()[]{}@#")
    itos = specials + lowers + uppers + digits + symbols
    stoi = {ch: i for i, ch in enumerate(itos)}
    return itos, stoi

ITOS, STOI = build_vocab()
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2

def encode_str(s: str, max_len: int) -> List[int]:
    s = s[:max_len - 2]  # reserve for BOS/EOS
    ids = [BOS_ID] + [STOI.get(ch, STOI.get('?', 0)) for ch in s] + [EOS_ID]
    return ids

def decode_ids(ids: list) -> str:
    out = []
    for t in ids:
        if t == EOS_ID:
            break
        if t in (PAD_ID, BOS_ID):
            continue
        out.append(ITOS[t])
    return ''.join(out)
