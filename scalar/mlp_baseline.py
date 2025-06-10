import os
import numpy as np
import pandas as pd
import argparse
import joblib
import copy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split

def load_data(processed_dir):
    X = pd.read_csv(os.path.join(processed_dir, 'X.csv')).values
    Y = pd.read_csv(os.path.join(processed_dir, 'Y.csv')).values
    return X, Y


def get_splits(X, Y, test_frac=0.1, random_state=42):
    # Reproduce cINN splitting: train/val/test
    n = X.shape[0]
    idx = np.arange(n)
    idx_tmp, idx_te = train_test_split(idx, test_size=test_frac, random_state=random_state)
    val_frac = test_frac / (1 - test_frac)
    idx_tr, idx_va = train_test_split(idx_tmp, test_size=val_frac, random_state=random_state)

    X_tr, Y_tr = X[idx_tr], Y[idx_tr]
    X_va, Y_va = X[idx_va], Y[idx_va]
    X_te, Y_te = X[idx_te], Y[idx_te]
    return X_tr, Y_tr, X_va, Y_va, X_te, Y_te


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
    best_loss = float('inf')
    epochs_no_improve = 0
    best_state = None
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        val_sum = 0.0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in loader_va:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                batch_loss = criterion(pred, yb).item() * yb.size(0)
                val_sum += batch_loss
                n_samples += yb.size(0)
        val_loss = val_sum / n_samples
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    model.load_state_dict(best_state)
    return model


def main(processed_dir,
         n_hidden_layers=3,
         hidden_units=256,
         batch_size=512,
         patience=20,
         ensemble_size=7,
         epochs_phase1=100,
         epochs_phase2=50):
    # Load data
    X, Y = load_data(processed_dir)
    # Splits
    X_tr, Y_tr, X_va, Y_va, X_te, Y_te = get_splits(X, Y)
    # DataLoaders
    def make_loader(Xa, Ya, shuffle):
        ds = TensorDataset(torch.tensor(Xa, dtype=torch.float32),
                           torch.tensor(Ya, dtype=torch.float32))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    loader_tr = make_loader(X_tr, Y_tr, shuffle=True)
    loader_va = make_loader(X_va, Y_va, shuffle=False)
    loader_te = make_loader(X_te, Y_te, shuffle=False)

    input_dim = X_tr.shape[1]
    output_dim = Y_tr.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load target scaler for inverse transform
    tar_scaler = joblib.load(os.path.join(processed_dir, 'tar_scaler.pkl'))

    # Ensemble
    all_preds = []
    for i in range(ensemble_size):
        model = MLP(input_dim, output_dim, n_hidden_layers, hidden_units).to(device)
        # Phase 1: MSE
        optimizer = Adam(model.parameters(), lr=1e-3)
        model = train_phase(model, loader_tr, loader_va, nn.MSELoss(), optimizer, device,
                             max_epochs=epochs_phase1, patience=patience)
        # Phase 2: MAE
        optimizer = Adam(model.parameters(), lr=5e-4)
        model = train_phase(model, loader_tr, loader_va, nn.L1Loss(), optimizer, device,
                             max_epochs=epochs_phase2, patience=patience)
        # Predict test
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader_te:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
        all_preds.append(np.vstack(preds))
        # Optionally save model
        torch.save(model.state_dict(), os.path.join(processed_dir, f'mlp_{i}.pt'))
    # Median ensemble
    ensemble_pred = np.median(np.stack(all_preds, axis=0), axis=0)
    # Inverse transform
    pred_phys = tar_scaler.inverse_transform(ensemble_pred)
    true_phys = tar_scaler.inverse_transform(Y_te)
    # Compute MAE per output
    mae = np.mean(np.abs(pred_phys - true_phys), axis=0)
    for j, val in enumerate(mae):
        print(f"Test MAE for output {j}: {val:.4f}")

    # Save results
    np.save(os.path.join(processed_dir, 'mlp_test_pred.npy'), pred_phys)
    np.save(os.path.join(processed_dir, 'mlp_test_true.npy'), true_phys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', type=str, default='processed_data')
    parser.add_argument('--hidden_layers', type=int, default=3)
    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--ensemble_size', type=int, default=7)
    parser.add_argument('--epochs_phase1', type=int, default=100)
    parser.add_argument('--epochs_phase2', type=int, default=50)
    args = parser.parse_args()
    main(
        processed_dir=args.processed_dir,
        n_hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        batch_size=args.batch_size,
        patience=args.patience,
        ensemble_size=args.ensemble_size,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2
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