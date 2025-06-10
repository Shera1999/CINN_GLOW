#!/usr/bin/env python3
import os
import re
import copy
import joblib
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ——————————————————————————————————————————————
# 1) Utilities to assemble combined data (3 proj per halo)
# ——————————————————————————————————————————————

def normalize_key(fname):
    # snap_050_halo_123456_proj_1.png → "123456_50"
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r"^snap_(\d+)_halo_(\d+)_proj_\d+$", base)
    return f"{int(m.group(2))}_{int(m.group(1))}"

def load_combined(processed_dir, embeddings_npy, filenames_npy):
    # load CSV‐scaled scalars + targets + meta
    dfX  = pd.read_csv(os.path.join(processed_dir, "X.csv"))
    dfY  = pd.read_csv(os.path.join(processed_dir, "Y.csv"))
    meta = pd.read_csv(os.path.join(processed_dir, "meta.csv"))
    X_s_all = dfX.values    # (2183, D_obs)
    Y_all   = dfY.values    # (2183, D_tar)

    # map halo_snap → row index
    key2idx = {
        f"{int(r.HaloID)}_{int(r.Snapshot)}": i
        for i, r in meta.iterrows()
    }

    # load all embeddings + filenames, filter to kept halos
    all_emb    = np.load(embeddings_npy)   # (6549, D_emb_total)
    all_files  = np.load(filenames_npy)    # (6549,)
    proj_keys, Es = [], []
    for fn, emb in zip(all_files, all_emb):
        key = normalize_key(fn)
        if key in key2idx:
            proj_keys.append(key)
            Es.append(emb)
    E_proj_raw = np.vstack(Es)             # (6549, D_emb)
    # assemble projection‐level scalars + targets
    X_s_proj = np.vstack([ X_s_all[key2idx[k]] for k in proj_keys ])
    Y_proj   = np.vstack([   Y_all[key2idx[k]] for k in proj_keys ])

    # scale embeddings same as training
    emb_sc = StandardScaler().fit(E_proj_raw)
    E_proj = emb_sc.transform(E_proj_raw)
    joblib.dump(emb_sc, os.path.join(processed_dir, "emb_scaler.pkl"))

    # combine
    X_all = np.hstack([X_s_proj, E_proj])  # (6549, D_obs+D_emb)
    Y_all = Y_proj                         # (6549, D_tar)
    return X_all, Y_all

def get_splits(X, Y, test_frac=0.1, random_state=42):
    n = X.shape[0]
    idx = np.arange(n)
    idx_tmp, idx_te = train_test_split(idx, test_size=test_frac, random_state=random_state)
    val_frac = test_frac / (1 - test_frac)
    idx_tr, idx_va = train_test_split(idx_tmp, test_size=val_frac, random_state=random_state)
    return (X[idx_tr], Y[idx_tr],
            X[idx_va], Y[idx_va],
            X[idx_te], Y[idx_te])

# ——————————————————————————————————————————————
# 2) MLP & training utilities (same as before)
# ——————————————————————————————————————————————

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden_layers, hidden_units):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_units, bias=False))
            layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.ReLU())
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_phase(model, loader_tr, loader_va, criterion, optimizer, device, max_epochs, patience):
    best_loss, epochs_no_improve, best_state = float('inf'), 0, None
    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        # validation
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader_va:
                xb, yb = xb.to(device), yb.to(device)
                batch_loss = criterion(model(xb), yb).item() * yb.size(0)
                total += batch_loss; count += yb.size(0)
        val_loss = total / count
        if val_loss < best_loss:
            best_loss = val_loss; epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    model.load_state_dict(best_state)
    return model

# ——————————————————————————————————————————————
# 3) Main: assemble, split, ensemble, predict & save
# ——————————————————————————————————————————————

def main(processed_dir,
         embeddings_npy, filenames_npy,
         n_hidden_layers=3,
         hidden_units=256,
         batch_size=512,
         patience=20,
         ensemble_size=7,
         epochs_phase1=100,
         epochs_phase2=50):
    # data
    X, Y = load_combined(processed_dir, embeddings_npy, filenames_npy)
    X_tr, Y_tr, X_va, Y_va, X_te, Y_te = get_splits(X, Y)

    # dataloaders
    def mk_loader(Xa, Ya, shuffle):
        ds = TensorDataset(torch.tensor(Xa, dtype=torch.float32),
                           torch.tensor(Ya, dtype=torch.float32))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    loader_tr = mk_loader(X_tr, Y_tr, True)
    loader_va = mk_loader(X_va, Y_va, False)
    loader_te = mk_loader(X_te, Y_te, False)

    input_dim, output_dim = X_tr.shape[1], Y_tr.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tar_scaler = joblib.load(os.path.join(processed_dir, 'tar_scaler.pkl'))

    all_preds = []
    for i in range(ensemble_size):
        model = MLP(input_dim, output_dim, n_hidden_layers, hidden_units).to(device)
        # phase 1 (MSE)
        opt = Adam(model.parameters(), lr=1e-3)
        model = train_phase(model, loader_tr, loader_va, nn.MSELoss(), opt, device, epochs_phase1, patience)
        # phase 2 (MAE)
        opt = Adam(model.parameters(), lr=5e-4)
        model = train_phase(model, loader_tr, loader_va, nn.L1Loss(), opt, device, epochs_phase2, patience)
        # predict
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader_te:
                preds.append(model(xb.to(device)).cpu().numpy())
        all_preds.append(np.vstack(preds))
        torch.save(model.state_dict(), os.path.join(processed_dir, f'mlp_comb_{i}.pt'))

    # ensemble median
    ensemble_pred = np.median(np.stack(all_preds,0), axis=0)
    pred_phys     = tar_scaler.inverse_transform(ensemble_pred)
    true_phys     = tar_scaler.inverse_transform(Y_te)

    # save
    np.save(os.path.join(processed_dir, 'mlp_test_pred_combined.npy'), pred_phys)
    np.save(os.path.join(processed_dir, 'mlp_test_true_combined.npy'), true_phys)

    # report MAE
    mae = np.mean(np.abs(pred_phys - true_phys), axis=0)
    cols = pd.read_csv(os.path.join(processed_dir, 'Y.csv')).columns
    for c,v in zip(cols, mae):
        print(f"Test MAE for {c}: {v:.4f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--processed_dir',  default='processed_data')
    p.add_argument('--embeddings_npy', default='embeddings.npy')
    p.add_argument('--filenames_npy',  default='filenames.npy')
    p.add_argument('--hidden_layers',  type=int, default=3)
    p.add_argument('--hidden_units',   type=int, default=256)
    p.add_argument('--batch_size',     type=int, default=512)
    p.add_argument('--patience',       type=int, default=20)
    p.add_argument('--ensemble_size',  type=int, default=7)
    p.add_argument('--epochs_phase1',  type=int, default=100)
    p.add_argument('--epochs_phase2',  type=int, default=50)
    args = p.parse_args()
    main(
      processed_dir  = args.processed_dir,
      embeddings_npy = args.embeddings_npy,
      filenames_npy  = args.filenames_npy,
      n_hidden_layers=args.hidden_layers,
      hidden_units   = args.hidden_units,
      batch_size     = args.batch_size,
      patience       = args.patience,
      ensemble_size  = args.ensemble_size,
      epochs_phase1  = args.epochs_phase1,
      epochs_phase2  = args.epochs_phase2
    )


# Build a sequence of Linear + BatchNorm + ReLU layers repeated n_hidden_layers times
# final layer is Linear mapping to output_dim
# rather than trianing once, we split into two phases:
# 1) Train MSE loss for n_epochs_phase1 with heigher learning rate: this penalize large mistajes more 
#    heavilythan small ones, so we can get a good initialization for the weights
#    optimizer = Adam(model.parameters(), lr=1e-3) and patience = 20
# 2) Train MAE loss for n_epochs_phase2: this penalizes all mistakes equally,
#    and yeilds better median type performance
#    optimizer = Adam(model.parameters(), lr=5e-4) and patience = 20

# by doing this:
# Phase 1) we get a good initialization for the weights and get into a good region of parameter space
# 2) we can train the last layer to be more robust to outliers and noisy samples, it fine-tunes the model

# Main workflow:
# 1) Load data and split into train/val/test
# 2) Build DataLoaders
# 3) Instantiate model and optimizer
# 4) Reload target scaler for inverse transform
# 5) ensemble loop: repeat the training process for ensemble_size 7 times:
#    - Inititate a fresh MLP with random weights
#    - Phase 1: train with MSE loss + lr 1e-3 with early stopping
#    - Phase 2: train with MAE loss + lr 5e-4 with early stopping
#    - Predict test set and collect raw (still scaled) predictions
#    - Save model
# 6) sketch the 7 sets of test predictions into a 7xNxD array and take the median to get single NxD prediction
# 7) Inverse transform the predictions and the median and ground truth to get the original physical units
# 8) Compute the mean absolute error (MAE) across all test samples for each output dimension
# 9) Save the predictions and true values to .npy files for later analysis