# train_cluster_cinn_combined.py

import os
import re
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from model import cINN

def nll_loss(z: torch.Tensor, log_jac: torch.Tensor) -> torch.Tensor:
    per_sample = 0.5 * (z**2).sum(dim=1) - log_jac
    return per_sample.mean()
# ————— utilities —————

def normalize_fname(fname: str) -> str:
    """Turn 'snap_050_halo_123456_proj_0' → '123456_50'."""
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r"^snap_(\d+)_halo_(\d+)_proj_\d+$", base)
    snap, halo = int(m.group(1)), int(m.group(2))
    return f"{halo}_{snap}"

def get_embeddings_for_meta(meta_df: pd.DataFrame,
                            emb_map: dict) -> np.ndarray:
    """Look up each (HaloID, Snapshot) in emb_map, stack into array."""
    missing, E = [], []
    for _, row in meta_df.iterrows():
        key = f"{int(row.HaloID)}_{int(row.Snapshot)}"
        if key not in emb_map:
            missing.append(key)
        else:
            E.append(emb_map[key])
    if missing:
        raise KeyError("Missing embeddings for keys: " + ", ".join(missing))
    return np.vstack(E)

def make_loader(X: np.ndarray, Y: np.ndarray,
                batch_size=512, shuffle=True):
    ds = TensorDataset(
        torch.tensor(Y, dtype=torch.float32),
        torch.tensor(X, dtype=torch.float32)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def main():
    processed_dir = "processed_data"
    # noise levels (you can tweak)
    obs_noise_std = 0.01
    emb_noise_std = 0.01
    patience      = 20

    # — 1) Load your scaled scalars + targets + meta —
    dfX  = pd.read_csv(os.path.join(processed_dir, "X.csv"))
    dfY  = pd.read_csv(os.path.join(processed_dir, "Y.csv"))
    meta = pd.read_csv(os.path.join(processed_dir, "meta.csv"))

    X_s_all = dfX.values      # (N, D_obs)
    Y_all   = dfY.values      # (N, D_tar)
    N, scalar_dim = X_s_all.shape
    _,         D_tar  = Y_all.shape

    # — 2) Load & map your raw embeddings (.npy) into emb_map —
    all_emb    = np.load("embeddings.npy")    # shape (N_proj, D_emb)
    all_fnames = np.load("filenames.npy")     # shape (N_proj,)
    df_emb     = pd.DataFrame(all_emb)
    df_emb["key"] = [normalize_fname(f) for f in all_fnames]
    emb_map    = {
        row["key"]: row.drop("key").values
        for _, row in df_emb.iterrows()
    }

    # — 3) Fetch & scale embeddings for each cluster in meta —
    E_all_raw = get_embeddings_for_meta(meta, emb_map)  # (N, D_emb)
    emb_sc     = StandardScaler().fit(E_all_raw)
    E_all      = emb_sc.transform(E_all_raw)
    # save for inference
    joblib.dump(emb_sc, os.path.join(processed_dir, "emb_scaler.pkl"))

    D_emb = E_all.shape[1]

    # — 4) Combine scalars + embeddings into one big condition vector —
    X_all = np.hstack([X_s_all, E_all])  # (N, scalar_dim + D_emb)
    total_x_dim = X_all.shape[1]

    # — 5) Train/val/test split (same seed for reproducibility) —
    idx       = np.arange(N)
    idx_tmp, idx_te = train_test_split(idx, test_size=0.1, random_state=42)
    val_frac  = 0.1 / 0.9
    idx_tr, idx_va = train_test_split(idx_tmp, test_size=val_frac, random_state=42)

    X_tr, Y_tr = X_all[idx_tr], Y_all[idx_tr]
    X_va, Y_va = X_all[idx_va], Y_all[idx_va]
    X_te, Y_te = X_all[idx_te], Y_all[idx_te]

    # — 6) DataLoaders —
    train_loader = make_loader(X_tr, Y_tr, batch_size=512, shuffle=True)
    val_loader   = make_loader(X_va, Y_va, batch_size=512, shuffle=False)

    # — 7) Build the combined‐condition cINN —
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print("Shapes  Train/Val/Test:", X_tr.shape, X_va.shape, X_te.shape)
    model  = cINN(y_dim=D_tar,
                  x_dim=total_x_dim,
                  hidden_dim=128,
                  n_blocks=12,
                  clamp=2.0).to(device)

    optimizer = Adam(model.parameters(),
                     lr=2e-4,
                     weight_decay=1e-4)

    # — 8) Training loop (noise on both parts) —
    best_val, epochs_without = float('inf'), 0
    for epoch in range(1, 251):
        # — train —
        model.train()
        train_losses = []
        for yb, xb in train_loader:
            yb, xb = yb.to(device), xb.to(device)
            x_s = xb[:, :scalar_dim]
            x_e = xb[:, scalar_dim:]
            # add independent noise
            x_s_n = x_s + torch.randn_like(x_s) * obs_noise_std
            x_e_n = x_e + torch.randn_like(x_e) * emb_noise_std
            x_n   = torch.cat([x_s_n, x_e_n], dim=1)

            z, log_jac = model(yb, x_n)
            loss = nll_loss(z, log_jac)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # — validate —
        model.eval()
        val_losses = []
        with torch.no_grad():
            for yb, xb in val_loader:
                yb, xb = yb.to(device), xb.to(device)
                x_s = xb[:, :scalar_dim]
                x_e = xb[:, scalar_dim:]
                # typically no noise at eval
                x_n = torch.cat([x_s, x_e], dim=1)
                z, log_jac = model(yb, x_n)
                val_losses.append(nll_loss(z, log_jac).item())

        t_nll = np.mean(train_losses)
        v_nll = np.mean(val_losses)
        print(f"Epoch {epoch:03d}  Train NLL: {t_nll:.4f}   Val NLL: {v_nll:.4f}")

        # — early stopping —
        if v_nll < best_val:
            best_val, epochs_without = v_nll, 0
            torch.save(model.state_dict(), 'best_cluster_cinn_combined.pt')
        else:
            epochs_without += 1
            if epochs_without >= patience:
                print(f"No improvement for {patience} epochs—stopping.")
                break

    print("Training complete. Best val NLL:", best_val)


if __name__ == "__main__":
    main()
