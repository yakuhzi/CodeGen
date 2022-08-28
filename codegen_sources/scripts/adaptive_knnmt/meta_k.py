from collections import OrderedDict
import math
from typing import List, Optional
from torch_scatter import scatter

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from codegen_sources.scripts.knnmt.knnmt import KNNMT


class MetaK(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        k: int,
        hidden_size: int,
        temperature: int,
        vocab_size: int,
        language_pair: str,
        beta_1: float = 0.9,
        beta_2: float = 0.98
    ):
        super(MetaK, self).__init__()
        self.save_hyperparameters()

        self.knnmt = KNNMT()

        self.sequential = nn.Sequential(
            nn.Linear(k * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_size, 2 + int(math.log(k, 2))),
            nn.Softmax(dim=-1) # [0 neighbor prob, 1 neighbor prob, 2 neighbor prob, 4 , 8 , ... , ]
        )

        nn.init.xavier_normal_(self.sequential[0].weight[:, :k], gain=0.01)
        nn.init.xavier_normal_(self.sequential[0].weight[:, k:], gain=0.1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        knns, distances = self.knnmt.get_k_nearest_neighbors(features, self.hparams.language_pair, k=self.hparams.k)
        distinct_neighbors = [[len(set(indices[:i])) for i in range(1, self.hparams.k + 1)] for indices in knns]

        distances = torch.FloatTensor(distances).to(self.device)
        distinct_neighbors = torch.FloatTensor(distinct_neighbors).to(self.device)

        input = torch.cat((distances, distinct_neighbors), dim=-1) # [B, 2 * K]
        k_prob = self.sequential(input) # [B, 1 + R_K] -> [B, (1 - lamda) + R_K] -> R_K indicates probabilities for k = [1 2 4 8 16 32]

        knn_lambda = 1. - k_prob[:, :1]  # [B, 1]
        k_soft_prob = k_prob[:, 1:] # [B, R_K]

        B, K = distances.size()
        R_K = k_soft_prob.size(-1)

        knn_tgt_prob = torch.zeros(B, self.hparams.vocab_size).to(self.device) # [B, Vocab Size]

        for i in range(R_K):
            k = pow(2, i) # [1 2 4 8 16 32]

            distances_i = distances[:, :k] # [B, k]
            normalized_distances = torch.softmax(distances_i / self.hparams.temperature * -1, dim=-1) # [B, k]

            prob = k_soft_prob[:, i] # [B]

            for batch_index, p in enumerate(prob):
                scores = normalized_distances[batch_index] * p # [k]

                for index, score in enumerate(scores):
                    tgt_index = knns[batch_index][index]
                    knn_tgt_prob[batch_index][tgt_index] += score
        
        # distances = distances.unsqueeze(-2).expand(B, R_K, K) # [B, R_K, K]
        # distances = distances / self.hparams.temperature # [B, R_K, K]

        # knn_weight = torch.softmax(distances, dim=-1) # [B, R_K, K]
        # weight_sum_knn_weight = torch.matmul(k_soft_prob.unsqueeze(-2), knn_weight).squeeze(-2).unsqueeze(-1) # [B, K, 1]

        # knn_tgt_prob = torch.zeros(B, K, self.hparams.vocab_size).to(device) # [B, K, Vocab Size]
        # tgt_index = tgt_index.unsqueeze_(-1) # [B, S, K, 1]

        # scatter(src=weight_sum_knn_weight.float(), out=knn_tgt_prob, index=tgt_index, dim=-1)
        # prob = knn_tgt_prob.sum(dim=-2) # [B, Vocab Size]

        return knn_lambda, knn_tgt_prob

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[OrderedDict]:
        loss = self.calculate_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss = self.calculate_loss(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss = self.calculate_loss(batch)
        self.log('test_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def calculate_loss(self, batch: torch.Tensor):
        features, tc_scores, targets = batch

        knn_lambda, knn_tgt_prob = self(features) # [B, 1]

        normalized_tc_scores = F.softmax(tc_scores.float(), dim=-1)
        y_hat = normalized_tc_scores * (1 - knn_lambda) + knn_tgt_prob
        y_hat = torch.clamp(y_hat, min=0, max=1)

        B = features.size()[0]

        y_star = torch.zeros(B, self.hparams.vocab_size).to(self.device)
        
        for index, target in enumerate(targets):
            y_star[index][target] = 1

        return F.binary_cross_entropy(y_hat, y_star)

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta_1, self.hparams.beta_2),
        )