#!/usr/bin/env python3
"""
Step 5: train the shared-backbone multi-task network.

Architecture
------------
                    ESM-2 embedding (1280-d, frozen)
                                |
                     shared backbone (MLP)
                    1280 -> 512 -> 256, LayerNorm + GELU + dropout
                                |
        +---------------+-------+-------+---------------+
        |               |               |               |
     cold head       heat head    drought head      salt head
     (256 -> 1)      (256 -> 1)    (256 -> 1)      (256 -> 1)
        |               |               |               |
     sigmoid         sigmoid         sigmoid         sigmoid

The heads are four INDEPENDENT SIGMOIDS, not a softmax. This is the specific
point on which the reference paper rejected a unified model: it argued that a
multiclass model's probabilities "would equal 1", so only the single
highest-scoring stress could be predicted. That objection applies to softmax
only. Independent sigmoids emit independent Bernoulli probabilities, so a gene
may legitimately score high for drought AND salt at once -- which is what the
239 drought+salt genes in this dataset actually are.

Class imbalance is handled with per-head pos_weight in the BCE loss rather
than resampling, so no data is discarded.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    EMB_DIR, SPLIT_DIR, MODELS_DIR, TABLES_DIR, STRESSES, RANDOM_STATE,
)


class MultiTaskNet(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256), dropout=0.3, n_tasks=4):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.backbone = nn.Sequential(*layers)
        # one independent head per stress
        self.heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_tasks)])

    def forward(self, x):
        z = self.backbone(x)
        return torch.cat([h(z) for h in self.heads], dim=1)  # logits


def evaluate(model, X, Y, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        probs = torch.sigmoid(logits).cpu().numpy()
    out = {}
    for i, s in enumerate(STRESSES):
        if len(np.unique(Y[:, i])) < 2:
            out[s] = {"auROC": float("nan"), "auPRC": float("nan")}
            continue
        out[s] = {
            "auROC": roc_auc_score(Y[:, i], probs[:, i]),
            "auPRC": average_precision_score(Y[:, i], probs[:, i]),
        }
    return out, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default=None,
                    help="path to .npz from 23_extract_esm_embeddings.py "
                         "(default: newest dataset_*.npz)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--hidden", default="512,256")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--tag", default="esm_multitask")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # ---- load embeddings + labels ----
    emb_path = args.embeddings
    if emb_path is None:
        cands = [f for f in os.listdir(EMB_DIR)
                 if f.startswith("dataset_") and f.endswith(".npz")]
        if not cands:
            sys.exit("ERROR: no dataset embeddings found -- run 23_extract_esm_embeddings.py")
        emb_path = os.path.join(EMB_DIR, sorted(cands)[-1])
    print(f"Embeddings: {emb_path}")
    z = np.load(emb_path, allow_pickle=True)
    id_to_vec = {i: v for i, v in zip(z["ids"].tolist(), z["emb"])}

    split = pd.read_csv(os.path.join(SPLIT_DIR, "split_assignments.tsv"), sep="\t")
    split = split[split["ProteinID"].isin(id_to_vec)].reset_index(drop=True)

    def pack(sub):
        X = np.stack([id_to_vec[p] for p in sub["ProteinID"]])
        Y = sub[STRESSES].to_numpy().astype(np.float32)
        return X, Y

    tr_all = split[split.split == "train"].reset_index(drop=True)
    te = split[split.split == "test"].reset_index(drop=True)

    # carve a validation set out of train for early stopping
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(len(tr_all))
    n_val = int(len(idx) * args.val_fraction)
    va = tr_all.iloc[idx[:n_val]].reset_index(drop=True)
    tr = tr_all.iloc[idx[n_val:]].reset_index(drop=True)

    Xtr, Ytr = pack(tr)
    Xva, Yva = pack(va)
    Xte, Yte = pack(te)

    # standardise using training statistics only
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

    print(f"train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}  dim {Xtr.shape[1]}")
    print("positives per head (train):",
          {s: int(Ytr[:, i].sum()) for i, s in enumerate(STRESSES)})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden = tuple(int(h) for h in args.hidden.split(","))
    model = MultiTaskNet(Xtr.shape[1], hidden, args.dropout).to(device)

    # per-head pos_weight = negatives/positives, so no data need be discarded
    pos = Ytr.sum(0)
    pos_weight = torch.tensor((len(Ytr) - pos) / np.maximum(pos, 1),
                              dtype=torch.float32, device=device)
    print("pos_weight per head:", {s: round(float(pos_weight[i]), 2)
                                   for i, s in enumerate(STRESSES)})
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32, device=device)

    best_score, best_state, bad, history = -np.inf, None, 0, []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        total = 0.0
        for b in range(0, len(perm), args.batch_size):
            sel = perm[b:b + args.batch_size]
            opt.zero_grad()
            loss = criterion(model(Xtr_t[sel]), Ytr_t[sel])
            loss.backward()
            opt.step()
            total += loss.item() * len(sel)
        sched.step()

        val_metrics, _ = evaluate(model, Xva, Yva, device)
        val_auprc = float(np.nanmean([val_metrics[s]["auPRC"] for s in STRESSES]))
        val_auroc = float(np.nanmean([val_metrics[s]["auROC"] for s in STRESSES]))
        history.append({"epoch": epoch + 1, "train_loss": total / len(Xtr_t),
                        "val_auROC": val_auroc, "val_auPRC": val_auprc})

        if val_auprc > best_score:
            best_score, bad = val_auprc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}  loss {total/len(Xtr_t):.4f}  "
                  f"val auROC {val_auroc:.4f}  val auPRC {val_auprc:.4f}")
        if bad >= args.patience:
            print(f"  early stop at epoch {epoch+1} (no val gain for {args.patience})")
            break

    model.load_state_dict(best_state)
    test_metrics, test_probs = evaluate(model, Xte, Yte, device)

    # auPRC is base-rate dependent: with ~2% positives a random model scores
    # ~2%, so absolute auPRC is NOT comparable to the single-task pipeline
    # (~26% positives). Report prevalence and lift over it as well.
    print("\n=== Held-out test performance (multi-task) ===")
    print(f"  {'stress':8s} {'auROC':>7s} {'auPRC':>7s} {'prev.':>7s} {'lift':>6s}")
    rows = []
    for i, s in enumerate(STRESSES):
        m = test_metrics[s]
        prev = float(Yte[:, i].mean())
        lift = m["auPRC"] / prev if prev > 0 else float("nan")
        print(f"  {s:8s} {m['auROC']*100:7.2f} {m['auPRC']*100:7.2f} "
              f"{prev*100:6.2f}% {lift:5.1f}x")
        rows.append({"stress": s, "auROC": round(m["auROC"] * 100, 2),
                     "auPRC": round(m["auPRC"] * 100, 2),
                     "positive_rate": round(prev * 100, 2),
                     "auPRC_lift_over_random": round(lift, 2)})
    pd.DataFrame(rows).to_csv(
        os.path.join(TABLES_DIR, f"{args.tag}_test_metrics.tsv"), sep="\t", index=False)
    pd.DataFrame(history).to_csv(
        os.path.join(TABLES_DIR, f"{args.tag}_training_history.tsv"), sep="\t", index=False)

    pred = pd.DataFrame(test_probs, columns=[f"{s}_prob" for s in STRESSES])
    pred.insert(0, "ProteinID", te["ProteinID"].values)
    for s in STRESSES:
        pred[f"{s}_true"] = te[s].values
    pred.to_csv(os.path.join(TABLES_DIR, f"{args.tag}_test_predictions.tsv"),
                sep="\t", index=False)

    torch.save({"state_dict": best_state,
                "in_dim": Xtr.shape[1], "hidden": hidden, "dropout": args.dropout,
                "mu": mu, "sd": sd, "stresses": STRESSES,
                "embeddings": os.path.basename(emb_path)},
               os.path.join(MODELS_DIR, f"{args.tag}.pt"))
    with open(os.path.join(MODELS_DIR, f"{args.tag}_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nModel -> {os.path.join(MODELS_DIR, args.tag + '.pt')}")
    print("Next: python3 25_evaluate_and_compare.py")


if __name__ == "__main__":
    main()
