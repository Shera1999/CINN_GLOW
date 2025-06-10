# train_cinn.py
import os
import joblib
import numpy   as np
import torch
from torch.utils.data     import TensorDataset, DataLoader
from torch.optim         import Adam
from sklearn.model_selection import train_test_split

from model import cINN
"""
def load_processed_data(processed_dir="processed_data",
                        test_size=0.1,
                        val_size=0.1,
                        random_state=42):
    # load CSVs
    import pandas as pd
    dfX = pd.read_csv(os.path.join(processed_dir, "X.csv"))
    dfY = pd.read_csv(os.path.join(processed_dir, "Y.csv"))
    X   = dfX.values
    Y   = dfY.values

    # load scalers
    obs_sc = joblib.load(os.path.join(processed_dir, "obs_scaler.pkl"))
    tar_sc = joblib.load(os.path.join(processed_dir, "tar_scaler.pkl"))

    # re‐scale
    Xs = obs_sc.transform(X)
    Ys = tar_sc.transform(Y)

    # train/val/test split
    X_tmp, X_te, Y_tmp, Y_te = train_test_split(
        Xs, Ys, test_size=test_size, random_state=random_state
    )
    val_frac = val_size / (1 - test_size)
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X_tmp, Y_tmp, test_size=val_frac, random_state=random_state
    )
    return X_tr, Y_tr, X_va, Y_va, X_te, Y_te, obs_sc, tar_sc
"""



def load_processed_data(processed_dir='processed_data',
                        test_size=0.1, val_size=0.1, random_state=42):
    """
    Load X.csv, Y.csv + their scalers from `processed_dir`, 
    then split into train/val/test.
    
    Returns
    -------
    X_tr, Y_tr, X_va, Y_va, X_te, Y_te : numpy arrays
    obs_sc, tar_sc                    : fitted StandardScalers
    """
    import pandas as pd

    # 1) Read the scaled data
    dfX = pd.read_csv(os.path.join(processed_dir, 'X.csv'))
    dfY = pd.read_csv(os.path.join(processed_dir, 'Y.csv'))
    X   = dfX.values
    Y   = dfY.values

    # 2) Load the scalers
    obs_sc = joblib.load(os.path.join(processed_dir, 'obs_scaler.pkl'))
    tar_sc = joblib.load(os.path.join(processed_dir, 'tar_scaler.pkl'))

    # 3) Split into train / hold‐out test
    X_tmp, X_te, Y_tmp, Y_te = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    # 4) Further split the remainder into train / val
    val_frac = val_size / (1 - test_size)
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X_tmp, Y_tmp, test_size=val_frac, random_state=random_state
    )

    return X_tr, Y_tr, X_va, Y_va, X_te, Y_te, obs_sc, tar_sc

def nll_loss(z: torch.Tensor, log_jac: torch.Tensor) -> torch.Tensor:
    """
    Negative log‐likelihood under standard normal prior on z:
        0.5 * ||z||^2  -  log|det J|
    averaged over the batch.
    """
    # (batch_size,)
    per_sample = 0.5 * (z**2).sum(dim=1) - log_jac
    return per_sample.mean()

def main():
    # -- 1) Load & split processed data --
    X_tr, Y_tr, X_va, Y_va, X_te, Y_te, obs_sc, tar_sc = \
        load_processed_data('processed_data')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print("Shapes  Train/Val/Test:", X_tr.shape, X_va.shape, X_te.shape)

    # -- 2) Build DataLoaders; note we feed (Y, X) since we map y|x → z --
    def make_loader(X, Y, batch_size=512, shuffle=True):
        ds = TensorDataset(
            torch.tensor(Y, dtype=torch.float32),
            torch.tensor(X, dtype=torch.float32)
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_tr, Y_tr, shuffle=True)
    val_loader   = make_loader(X_va, Y_va, shuffle=False)

    # -- 3) Instantiate model & optimizer --
    y_dim, x_dim = Y_tr.shape[1], X_tr.shape[1]
    model = cINN(y_dim=y_dim,
                 x_dim=x_dim,
                 hidden_dim=128,
                 n_blocks=12,
                 clamp=2.0).to(device)
    optimizer = Adam(model.parameters(), lr=2e-3)

    best_val = float('inf')

    # -- 4) Training loop --
    for epoch in range(1, 251):
        # ---- train ----
        model.train()
        train_losses = []
        for yb, xb in train_loader:
            yb, xb = yb.to(device), xb.to(device)
            z, log_jac = model(yb, xb)
            loss = nll_loss(z, log_jac)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ---- validate ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for yb, xb in val_loader:
                yb, xb = yb.to(device), xb.to(device)
                z, log_jac = model(yb, xb)
                val_losses.append(nll_loss(z, log_jac).item())

        t_nll = np.mean(train_losses)
        v_nll = np.mean(val_losses)
        print(f"Epoch {epoch:02d}  Train NLL: {t_nll:.4f}   Val NLL: {v_nll:.4f}")

        # ---- checkpoint best ----
        if v_nll < best_val:
            best_val = v_nll
            torch.save(model.state_dict(), 'best_cluster_cinn.pt')

    print("Training complete. Best val NLL:", best_val)

if __name__ == '__main__':
    main()

# load_processed_data
# Reads your cleaned & scaled CSVs (X.csv,Y.csv), reloads the obs_scaler.pkl and tar_scaler.pkl, then splits into train/validation/test.

#nll_loss(z, log_jac)
#Computes the negative log‐likelihood under a standard normal prior on z, averaged over the batch.

#main()

#1. Loads & splits data, moves to GPU if available.
#2. Wraps data in PyTorch DataLoaders — note we feed (Y, X) because our flow models y|x → z.
#3. Instantiates the cINN with your hyperparameters.
#4. Runs a standard training loop for 50 epochs, tracking both train and validation NLL.
#5. Checkpoints the model when validation NLL improves.
#6. At the end, prints out the best validation NLL and saves the final best_cluster_cinn.pt.