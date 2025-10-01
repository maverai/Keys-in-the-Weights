from copy import deepcopy
from typing import Dict, Tuple
import torch
from .config import DEVICE, ModelConfig, TrainConfig
from .vocab import ITOS, decode_ids, PAD_ID
from .data import make_loaders
from .model import TransformerAutoencoder
from .train import train_one
from .utils import set_seed, trim_to_eos, levenshtein, weight_l2_distance, attention_divergence

@torch.no_grad()
def evaluate_pairs(models: Dict[str, TransformerAutoencoder],
                   loader,
                   n_batches: int = 10,
                   max_decode_len: int = 50) -> Dict[Tuple[str, str], Dict[str, float]]:
    keys = list(models.keys())
    results = {}
    batches = []
    for i, (src, tgt) in enumerate(loader):
        batches.append((src.to(DEVICE), tgt.to(DEVICE)))
        if i+1 >= n_batches:
            break

    for ka in keys:
        A = models[ka]
        A.eval()
        for kb in keys:
            B = models[kb]
            B.eval()
            exact, tok_correct, tok_total = 0, 0, 0
            lev_sum, seqs = 0.0, 0
            for src, tgt in batches:
                mem, src_key_pad, _ = A.encode(src, need_attn=False)
                y = B.decode_with_external_memory(mem, src_key_pad, max_len=max_decode_len, need_attn=False)
                for i in range(src.size(0)):
                    ref = tgt[i].tolist()
                    hyp = y[i].tolist()
                    ref_str = decode_ids(ref)
                    hyp_str = decode_ids(hyp)
                    if ref_str == hyp_str:
                        exact += 1
                    r_trim = trim_to_eos(ref[1:])
                    h_trim = trim_to_eos(hyp[1:])
                    L = max(len(r_trim), len(h_trim))
                    r_pad = r_trim + [PAD_ID]*(L - len(r_trim))
                    h_pad = h_trim + [PAD_ID]*(L - len(h_trim))
                    tok_correct += sum(1 for a,b in zip(r_pad, h_pad) if a == b)
                    tok_total += L
                    lev_sum += 1.0 - (levenshtein(ref_str, hyp_str) /
                                      max(1, max(len(ref_str), len(hyp_str))))
                    seqs += 1
            results[(ka, kb)] = {
                "exact_match_pct": 100.0 * exact / (seqs if seqs else 1),
                "token_acc_pct": 100.0 * tok_correct / (tok_total if tok_total else 1),
                "levenshtein_sim_pct": 100.0 * lev_sum / (seqs if seqs else 1),
            }
            print(f"[{ka} enc → {kb} dec] exact={results[(ka,kb)]['exact_match_pct']:.2f}% "
                  f"tok_acc={results[(ka,kb)]['token_acc_pct']:.2f}% "
                  f"levsim={results[(ka,kb)]['levenshtein_sim_pct']:.2f}%")
    return results

def main():
    cfg_m = ModelConfig(vocab_size=len(ITOS))
    cfg_t = TrainConfig()
    MAX_LEN = cfg_m.max_len

    train_loader, test_loader = make_loaders(cfg_t.batch_size, MAX_LEN)

    print("\n==> Train Model-1 (seed=111)")
    set_seed(111, deterministic=False)
    model1 = TransformerAutoencoder(cfg_m).to(DEVICE)
    train_one(model1, train_loader, cfg_t, MAX_LEN)

    print("\n==> Train Model-2 (seed=222)")
    set_seed(222, deterministic=False)
    model2 = TransformerAutoencoder(cfg_m).to(DEVICE)
    train_one(model2, train_loader, cfg_t, MAX_LEN)

    print("\n==> Train Model-3 (seed=333)")
    set_seed(333, deterministic=False)
    model3 = TransformerAutoencoder(cfg_m).to(DEVICE)
    train_one(model3, train_loader, cfg_t, MAX_LEN)

    print("\n==> Make Model-1-CLONE (identical weights)")
    model1_clone = deepcopy(model1).to(DEVICE)

    print("\n==> Train Model-1-sameseed (seed=111 again)")
    set_seed(111, deterministic=False)
    model1_sameseed = TransformerAutoencoder(cfg_m).to(DEVICE)
    train_one(model1_sameseed, train_loader, cfg_t, MAX_LEN)

    print("\n==> Weight L2 distances (sqrt(sum((W1-W2)^2)))")
    d12 = weight_l2_distance(model1, model2)
    d13 = weight_l2_distance(model1, model3)
    d11c = weight_l2_distance(model1, model1_clone)
    d11s = weight_l2_distance(model1, model1_sameseed)
    print(f"dist(Model1, Model2)           = {d12:.4f}")
    print(f"dist(Model1, Model3)           = {d13:.4f}")
    print(f"dist(Model1, Model1_CLONE)     = {d11c:.4f}  (should be ~0)")
    print(f"dist(Model1, Model1_SAMESEED)  = {d11s:.4f}  (0 only if truly deterministic)")

    print("\n==> Evaluate self-decoding & cross-decoding")
    models = {
        "M1": model1,
        "M2": model2,
        "M3": model3,
        "M1_CLONE": model1_clone,
        "M1_SAMESEED": model1_sameseed
    }
    results = evaluate_pairs(models, test_loader, n_batches=6, max_decode_len=MAX_LEN)

    print("\n==> Summary (encoder → decoder)")
    for a,b in [("M1","M1"), ("M1","M2"), ("M1","M3"), ("M1","M1_CLONE"), ("M1","M1_SAMESEED")]:
        m = results[(a,b)]
        print(f"{a} → {b:12s} | exact={m['exact_match_pct']:.2f}% | "
              f"token={m['token_acc_pct']:.2f}% | lev-sim={m['levenshtein_sim_pct']:.2f}%")

    print("\n==> Attention divergence (encoder layer-0, head-avg)")
    for a,b in [("M1","M2"), ("M1","M3"), ("M1","M1_CLONE"), ("M1","M1_SAMESEED")]:
        met = attention_divergence(models[a], models[b], test_loader, n_batches=3)
        print(f"{a} vs {b}: KL(A||B)={met['encoder_layer0_headavg_KL_A||B']:.4f}, "
              f"cos={met['encoder_layer0_headavg_cosine']:.4f}")

if __name__ == "__main__":
    main()
