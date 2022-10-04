
import pytorch_lightning as pl
import torch
import math
import torch.nn.functional as F

from typing import Optional, Tuple
from collections import OrderedDict
from torch import nn
from codegen_sources.scripts.knnmt.knnmt import KNNMT

try:
    from torch_scatter import scatter
except ImportError:
    pass


class MetaK(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        batch_size: int,
        max_k: int,
        tc_k: int,
        hidden_size: int,
        knn_temperature: int,
        tc_temperature: int,
        vocab_size: int,
        language_pair: str,
        adam_betas: str,
        knnmt_dir: str,
    ):
        super(MetaK, self).__init__()
        self.save_hyperparameters()

        # Define kNN-MT datastore
        self.knnmt = KNNMT(knnmt_dir)

        # Define network layers
        self.sequential = nn.Sequential(
            nn.Linear(max_k * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2 + int(math.log(max_k, 2))),
            nn.Softmax(dim=-1)  # [0 neighbor prob, 1 neighbor prob, 2 neighbor prob, 4 , 8 , ... , ]
        )

        # Initialize network weights
        nn.init.xavier_normal_(self.sequential[0].weight[:, :max_k], gain=0.01)
        nn.init.xavier_normal_(self.sequential[0].weight[:, max_k:], gain=0.1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor. torch.Tensor]:
        # Get input features
        knns, distances = self.knnmt.get_k_nearest_neighbors(features, self.hparams.language_pair, k=self.hparams.max_k)
        distinct_neighbors = [[len(set(indices[:i])) for i in range(1, self.hparams.max_k + 1)] for indices in knns]

        # Convert features to tensors and put to GPU
        knns = torch.LongTensor(knns).to(self.device)
        distances = torch.FloatTensor(distances).to(self.device)
        distinct_neighbors = torch.FloatTensor(distinct_neighbors).to(self.device)

        # Concatenate features to input tensor
        input = torch.cat((distances, distinct_neighbors), dim=-1)  # [B, 2 * K]

        # Get network predictions
        k_prob = self.sequential(input)  # [B, 1 + R_K] -> [B, (1 - lambda) + R_K] -> R_K indicates probabilities for k = [1 2 4 8 16 32]

        # Separate lambda from k_prob
        knn_lambda = 1. - k_prob[:, :1]  # [B, 1]
        k_soft_prob = k_prob[:, 1:]  # [B, R_K]

        # Define target probabilities for each bach and word in vocab
        knn_tgt_prob = torch.zeros(distances.size(0), self.hparams.vocab_size).to(self.device)  # [B, Vocab Size]

        for i in range(k_soft_prob.size(-1)):
            k = pow(2, i)  # [1 2 4 8 16 32]

            # Get k distances
            distances_i = distances[:, :k]  # [B, k]
            # Normalize distances
            normalized_distances = torch.softmax(distances_i / self.hparams.knn_temperature * -1, dim=-1)  # [B, k]

            # Probability of k = i
            prob = k_soft_prob[:, i]  # [B]
            # Multiply distances with probability of k = i
            scores = (normalized_distances.T * prob).T  # [B, k]
            # Add scores to target probabilities
            scatter(src=scores, out=knn_tgt_prob, index=knns[:, :k], dim=-1)

        return knn_lambda, knn_tgt_prob

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.calculate_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.calculate_loss(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.calculate_loss(batch)
        self.log('test_loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def calculate_loss(self, batch: torch.Tensor) -> torch.Tensor:
        # Get network predictions
        features, tc_scores, targets, _, _ = batch
        # Combine network predictions with TransCoder predictions
        knn_tgt_prob = self.combined_prediction(features, tc_scores)
        # Clamp result in case of rounding errors and to avoid inf results for log operation
        knn_tgt_prob = torch.clamp(knn_tgt_prob, min=1e-10, max=1)
        return F.nll_loss(torch.log(knn_tgt_prob), targets)

    def combined_prediction(self, features, tc_scores) -> torch.Tensor:
        # Get network predictions
        knn_lambda, knn_tgt_prob = self(features)  # [B, 1], [B, R_K]
        # Get top k TransCoder predictions
        tc_topk = torch.topk(tc_scores, self.hparams.tc_k)
        # Normalize TransCoder scores
        normalized_tc_scores = F.softmax(tc_topk[0].float() / self.hparams.tc_temperature, dim=-1)
        # Add TransCoder scores to network predictions
        scatter(src=normalized_tc_scores * (1 - knn_lambda), out=knn_tgt_prob, index=tc_topk[1], dim=-1)
        return knn_tgt_prob

    def configure_optimizers(self) -> torch.optim.Adam:
        beta_1 = float(self.hparams.adam_betas.split(", ")[0])
        beta_2 = float(self.hparams.adam_betas.split(", ")[1])

        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(beta_1, beta_2),
        )
