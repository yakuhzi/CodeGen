
import pytorch_lightning as pl
import torch
import math
import torch.nn.functional as F

from typing import List, Optional
from torch_scatter import scatter
from collections import OrderedDict
from torch import nn
from codegen_sources.scripts.knnmt.knnmt import KNNMT


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

        self.knnmt = KNNMT(knnmt_dir)

        self.sequential = nn.Sequential(
            nn.Linear(max_k * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2 + int(math.log(max_k, 2))),
            nn.Softmax(dim=-1) # [0 neighbor prob, 1 neighbor prob, 2 neighbor prob, 4 , 8 , ... , ]
        )

        nn.init.xavier_normal_(self.sequential[0].weight[:, :max_k], gain=0.01)
        nn.init.xavier_normal_(self.sequential[0].weight[:, max_k:], gain=0.1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        knns, distances = self.knnmt.get_k_nearest_neighbors(features, self.hparams.language_pair, k=self.hparams.max_k)
        distinct_neighbors = [[len(set(indices[:i])) for i in range(1, self.hparams.max_k + 1)] for indices in knns]

        knns = torch.LongTensor(knns).to(self.device)
        distances = torch.FloatTensor(distances).to(self.device)
        distinct_neighbors = torch.FloatTensor(distinct_neighbors).to(self.device)

        input = torch.cat((distances, distinct_neighbors), dim=-1) # [B, 2 * K]
        k_prob = self.sequential(input) # [B, 1 + R_K] -> [B, (1 - lamda) + R_K] -> R_K indicates probabilities for k = [1 2 4 8 16 32]

        knn_lambda = 1. - k_prob[:, :1]  # [B, 1]
        k_soft_prob = k_prob[:, 1:] # [B, R_K]

        knn_tgt_prob = torch.zeros(distances.size(0), self.hparams.vocab_size).to(self.device) # [B, Vocab Size]

        for i in range(k_soft_prob.size(-1)):
            k = pow(2, i) # [1 2 4 8 16 32]

            distances_i = distances[:, :k] # [B, k]
            normalized_distances = torch.softmax(distances_i / self.hparams.knn_temperature * -1, dim=-1) # [B, k]

            prob = k_soft_prob[:, i] # [B]
            scores = (normalized_distances.T * prob).T # [B, k]
            scatter(src=scores, out=knn_tgt_prob, index=knns[:, :k], dim=-1)

        return knn_lambda, knn_tgt_prob

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[OrderedDict]:
        loss = self.calculate_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss = self.calculate_loss(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss = self.calculate_loss(batch)
        self.log('test_loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def calculate_loss(self, batch: torch.Tensor):
        features, tc_scores, targets, _, _ = batch
        knn_tgt_prob = self.combined_prediction(features, tc_scores)
        knn_tgt_prob = torch.clamp(knn_tgt_prob, min=1e-10, max=1)
        return F.nll_loss(torch.log(knn_tgt_prob), targets)

    def combined_prediction(self, features, tc_scores):
        knn_lambda, knn_tgt_prob = self(features) # [B, 1], [B, R_K]
        tc_topk = torch.topk(tc_scores, self.hparams.tc_k)
        normalized_tc_scores = F.softmax(tc_topk[0].float() / self.hparams.tc_temperature, dim=-1) 
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