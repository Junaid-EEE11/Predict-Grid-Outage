import logging
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from grid_outage_research.models.base import BaseOutageModel

logger = logging.getLogger(__name__)

class GRUNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class TemporalNNModel(BaseOutageModel):
    """Instant C-contiguous Causal Temporal GRU model."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 12,
        hidden_dim: int = 32,
        lr: float = 0.005,
        epochs: int = 5,
        batch_size: int = 512,
        device: str = "cpu"
    ):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.network = GRUNetwork(input_dim, hidden_dim).to(self.device)

    def _make_seqs(self, X: np.ndarray) -> np.ndarray:
        T, D = X.shape
        # Pure C-level sliding window
        windows = np.lib.stride_tricks.sliding_window_view(X, (self.seq_len, D))[:, 0, :, :].copy().astype(np.float32)
        pad = np.repeat(windows[:1], self.seq_len - 1, axis=0)
        return np.vstack([pad, windows])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TemporalNNModel":
        torch.set_num_threads(4)
        X_seq = self._make_seqs(X)
        y_seq = y.astype(np.float32)

        # Subsample 15,000 for 1-second training
        if len(X_seq) > 15000:
            idx = np.random.RandomState(42).choice(len(X_seq), size=15000, replace=False)
            X_tr, y_tr = X_seq[idx], y_seq[idx]
        else:
            X_tr, y_tr = X_seq, y_seq

        X_tensor = torch.from_numpy(X_tr)
        y_tensor = torch.from_numpy(y_tr).unsqueeze(1)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.network.train()
        n = len(X_tr)
        for epoch in range(self.epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, self.batch_size):
                b_idx = perm[i:i + self.batch_size]
                bx = X_tensor[b_idx].to(self.device)
                by = y_tensor[b_idx].to(self.device)

                optimizer.zero_grad()
                out = self.network(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch.set_num_threads(4)
        self.network.eval()
        X_seq = self._make_seqs(X)

        preds = []
        with torch.no_grad():
            for i in range(0, len(X_seq), 4096):
                batch = torch.from_numpy(X_seq[i:i+4096]).to(self.device)
                p = self.network(batch).cpu().numpy()
                preds.append(p)

        p1 = np.vstack(preds)
        p0 = 1.0 - p1
        return np.hstack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
