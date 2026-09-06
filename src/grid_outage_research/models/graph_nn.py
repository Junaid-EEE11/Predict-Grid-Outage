import logging
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from grid_outage_research.models.base import BaseOutageModel

logger = logging.getLogger(__name__)

class SpatiotemporalGNN(nn.Module):
    def __init__(self, num_nodes: int, input_dim: int, A_norm: torch.Tensor, hidden_dim: int = 24, use_graph: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.A_norm = A_norm
        self.use_graph = use_graph

        self.temporal_gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.spatial_linear = nn.Linear(hidden_dim, hidden_dim) if use_graph else nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, N, D)
        B, S, N, D = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, S, D)
        gru_out, _ = self.temporal_gru(x_reshaped)
        h_t = gru_out[:, -1, :].reshape(B, N, -1) # (B, N, hidden_dim)

        if self.use_graph:
            h_spatial = torch.einsum("ij,bjk->bik", self.A_norm, h_t)
            h_spatial = torch.relu(self.spatial_linear(h_spatial))
            out = self.fc(h_spatial)
        else:
            out = self.fc(h_t)

        return out

class SpatiotemporalGNNModel(BaseOutageModel):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        A_norm: np.ndarray,
        seq_len: int = 12,
        hidden_dim: int = 24,
        use_graph_passing: bool = True,
        lr: float = 0.005,
        epochs: int = 5,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.use_graph_passing = use_graph_passing
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.A_norm_tensor = torch.tensor(A_norm, dtype=torch.float32).to(self.device)

        self.network = SpatiotemporalGNN(
            num_nodes=num_nodes,
            input_dim=input_dim,
            A_norm=self.A_norm_tensor,
            hidden_dim=hidden_dim,
            use_graph=use_graph_passing
        ).to(self.device)

    def _make_3d_seqs(self, X_3d: np.ndarray) -> np.ndarray:
        T, N, D = X_3d.shape
        raw_windows = np.lib.stride_tricks.sliding_window_view(X_3d, (self.seq_len, N, D))
        # raw_windows shape is (T - seq_len + 1, 1, 1, seq_len, N, D)
        windows = raw_windows.reshape(-1, self.seq_len, N, D).copy().astype(np.float32)
        pad = np.repeat(windows[:1], self.seq_len - 1, axis=0)
        return np.vstack([pad, windows])

    def fit(self, X_3d: np.ndarray, y_2d: np.ndarray) -> "SpatiotemporalGNNModel":
        torch.set_num_threads(4)
        X_seq = self._make_3d_seqs(X_3d)
        y_seq = y_2d.astype(np.float32)

        if len(X_seq) > 2000:
            idx = np.random.RandomState(42).choice(len(X_seq), size=2000, replace=False)
            X_tr, y_tr = X_seq[idx], y_seq[idx]
        else:
            X_tr, y_tr = X_seq, y_seq

        X_tensor = torch.from_numpy(X_tr)
        y_tensor = torch.from_numpy(y_tr).unsqueeze(-1)

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

    def predict_proba(self, X_3d: np.ndarray) -> np.ndarray:
        torch.set_num_threads(4)
        self.network.eval()
        X_seq = self._make_3d_seqs(X_3d)

        preds = []
        with torch.no_grad():
            for i in range(0, len(X_seq), 256):
                batch = torch.from_numpy(X_seq[i:i+256]).to(self.device)
                p = self.network(batch).cpu().numpy()
                preds.append(p)

        # Output shape: (T * N, 1)
        p1 = np.vstack(preds).reshape(-1, 1)
        p0 = 1.0 - p1
        return np.hstack([p0, p1])

    def predict(self, X_3d: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X_3d)[:, 1] >= 0.5).astype(int)
